import os
import math
import time
import logging
from pathlib import Path
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from jiwer import wer

import sentencepiece as spm

# Import nos modules
from model import WhisperLike, ModelConfig
from dataset import SpeechDataset, get_dataloaders

# Logging
Path("logs").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/training.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# CONFIG ENTRAÎNEMENT
@dataclass
class TrainConfig:
    # Chemins
    train_manifest : str = "G:/Armand/New-Whisper/data/manifests/train.json"
    val_manifest   : str = "G:/Armand/New-Whisper/data/manifests/validation.json"
    tokenizer_path : str = "G:/Armand/New-Whisper/data/tokenizer/tokenizer_fr.model"
    save_dir       : str = "G:/Armand/New-Whisper/models"
    log_dir        : str = "G:/Armand/New-Whisper/logs/tensorboard"

    # Entraînement
    max_epochs     : int   = 50
    batch_size     : int   = 4      # batch réel sur GPU
    accum_steps    : int   = 16     # batch effectif = 4 × 16 = 64
    learning_rate  : float = 3e-4
    warmup_steps   : int   = 2000
    weight_decay   : float = 1e-2
    max_grad_norm  : float = 1.0
    patience       : int   = 5      # early stopping

    # Système
    num_workers    : int   = 0      # 0 = pas de multiprocessing (plus stable Windows)
    fp16           : bool  = True
    log_every      : int   = 50     # steps
    eval_every     : int   = 1      # époques
    save_every     : int   = 1      # époques


# SCHEDULER : WARMUP + COSINE DECAY
class WarmupCosineScheduler:
    """
    Learning rate schedule en deux phases :

    Phase 1 - Warmup linéaire (0 → max_lr sur warmup_steps)
      Le modèle démarre avec un LR très bas pour stabiliser
      les premiers updates. Monter trop vite → gradients explosent.

    Phase 2 - Décroissance cosinus (max_lr → ~0)
      Décroissance douce qui permet de converger finement
      vers un minimum sans osciller.
    """
    def __init__(self, optimizer, warmup_steps: int, total_steps: int):
        self.optimizer    = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.step_num     = 0
        self.base_lrs     = [pg["lr"] for pg in optimizer.param_groups]

    def step(self):
        self.step_num += 1
        lrs = self._get_lrs()
        for pg, lr in zip(self.optimizer.param_groups, lrs):
            pg["lr"] = lr

    def _get_lrs(self):
        s = self.step_num
        if s < self.warmup_steps:
            # Warmup linéaire
            factor = s / max(1, self.warmup_steps)
        else:
            # Cosine decay
            progress = (s - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            factor   = 0.5 * (1.0 + math.cos(math.pi * progress))
            factor   = max(factor, 1e-7 / self.base_lrs[0])
        return [base * factor for base in self.base_lrs]

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


# LOSS - CROSS ENTROPY AVEC PADDING IGNORÉ
def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pad_id: int,
) -> torch.Tensor:
    """
    CrossEntropyLoss position par position.

    logits  : [B, U, vocab_size] - prédictions du modèle
    targets : [B, U]             - vrais tokens (tgt_out)
    pad_id  : int                - positions à ignorer

    On aplatit tout en [B*U, vocab] et [B*U] pour
    que CrossEntropyLoss puisse calculer en une passe.

    ignore_index=pad_id : les positions padding ne contribuent
    pas à la loss - sinon le modèle serait pénalisé pour avoir
    prédit n'importe quoi sur des positions vides.
    """
    B, U, V = logits.shape

    # Aplatir [B, U, V] → [B*U, V] et [B, U] → [B*U]
    logits_flat  = logits.reshape(B * U, V)
    targets_flat = targets.reshape(B * U)

    loss = F.cross_entropy(
        logits_flat,
        targets_flat,
        ignore_index=pad_id,
        reduction="mean",
        label_smoothing=0.1,   # Label smoothing - évite la sur-confiance
    )
    return loss


# ÉVALUATION - WER
@torch.no_grad()
def evaluate(
    model: WhisperLike,
    val_loader,
    tokenizer,
    device: torch.device,
    cfg: TrainConfig,
    n_batches: int = 20,    # limiter pour aller vite
) -> tuple:
    """
    Évalue le modèle sur le val set.
    Calcule la loss ET le WER (Word Error Rate).

    Pour le WER, on génère le texte en mode autorégressif
    (pas de teacher forcing) - c'est comme en production.
    """
    model.eval()
    total_loss = 0.0
    n_loss     = 0

    pad_id = tokenizer.piece_to_id("<pad>")
    bos_id = tokenizer.piece_to_id("<s>")
    eos_id = tokenizer.piece_to_id("</s>")

    all_refs = []
    all_hyps = []

    for i, (mel, tgt_in, tgt_out, mel_len, tgt_len) in enumerate(val_loader):
        mel    = mel.to(device)
        tgt_in = tgt_in.to(device)
        tgt_out= tgt_out.to(device)
        mel_len= mel_len.to(device)

        # Loss avec teacher forcing
        with autocast("cuda", enabled=cfg.fp16):
            logits = model(mel, tgt_in, mel_len)
            loss   = compute_loss(logits, tgt_out, pad_id)

        total_loss += loss.item()
        n_loss     += 1

        # WER sur les N premiers batches seulement
        if i < n_batches:
            preds = model.generate(mel, mel_len, max_new_tokens=100, beam_size=1)

            for j, (pred_ids, tgt_ids, tl) in enumerate(
                zip(preds, tgt_out.cpu(), tgt_len.cpu())
            ):
                # Référence - retirer EOS et PAD
                ref_ids = tgt_ids[:tl].tolist()
                if eos_id in ref_ids:
                    ref_ids = ref_ids[:ref_ids.index(eos_id)]
                ref = tokenizer.decode(ref_ids).lower().strip()

                # Hypothèse
                hyp = tokenizer.decode(pred_ids).lower().strip()

                if ref:
                    all_refs.append(ref)
                    all_hyps.append(hyp)

    avg_loss = total_loss / max(n_loss, 1)
    val_wer  = wer(all_refs, all_hyps) if all_refs else 1.0

    # Afficher quelques exemples
    for ref, hyp in zip(all_refs[:3], all_hyps[:3]):
        log.info(f"  REF : {ref}")
        log.info(f"  HYP : {hyp}")
        log.info("")

    model.train()
    return avg_loss, val_wer


# BOUCLE D'ENTRAÎNEMENT
def train():
    cfg        = TrainConfig()
    model_cfg  = ModelConfig()
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info("=" * 55)
    log.info("Démarrage entraînement Encoder-Decoder")
    log.info("=" * 55)
    log.info(f"Device : {device}")
    if device.type == "cuda":
        log.info(f"GPU    : {torch.cuda.get_device_name(0)}")
        log.info(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Tokenizer
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(cfg.tokenizer_path)
    pad_id = tokenizer.piece_to_id("<pad>")
    bos_id = tokenizer.piece_to_id("<s>")
    eos_id = tokenizer.piece_to_id("</s>")

    # Mettre à jour vocab_size dans le config modèle
    model_cfg.vocab_size = tokenizer.get_piece_size()
    model_cfg.pad_id     = pad_id
    model_cfg.bos_id     = bos_id
    model_cfg.eos_id     = eos_id

    log.info(f"Tokenizer : {tokenizer.get_piece_size()} tokens")
    log.info(f"  pad={pad_id}, bos={bos_id}, eos={eos_id}")

    # DataLoaders
    log.info("\nChargement des datasets...")
    train_loader, val_loader = get_dataloaders(
        train_manifest=cfg.train_manifest,
        val_manifest=cfg.val_manifest,
        tokenizer_path=cfg.tokenizer_path,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
    )
    log.info(f"Train : {len(train_loader.dataset)} samples")
    log.info(f"Val   : {len(val_loader.dataset)} samples")

    # Modèle
    model  = WhisperLike(model_cfg).to(device)
    params = model.count_parameters()
    log.info(f"\nModèle : {params['total_M']}M paramètres")

    # Optimiseur & Scheduler
    optimizer    = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    total_steps  = len(train_loader) * cfg.max_epochs // cfg.accum_steps
    scheduler    = WarmupCosineScheduler(optimizer, cfg.warmup_steps, total_steps)
    scaler       = torch.amp.GradScaler("cuda", enabled=cfg.fp16)
    writer       = SummaryWriter(cfg.log_dir)

    # Reprise checkpoint
    save_dir     = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt  = save_dir / "latest.pt"

    start_epoch  = 0
    best_wer     = float("inf")
    patience_ctr = 0
    global_step  = 0

    if latest_ckpt.exists():
        log.info(f"\nReprise depuis {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch  = ckpt["epoch"] + 1
        best_wer     = ckpt.get("best_wer", float("inf"))
        global_step  = ckpt.get("global_step", 0)
        log.info(f"Reprise époque {start_epoch}, WER best={best_wer*100:.1f}%")

    # BOUCLE PRINCIPALE
    for epoch in range(start_epoch, cfg.max_epochs):
        model.train()
        epoch_loss  = 0.0
        epoch_start = time.time()
        optimizer.zero_grad()

        for step, (mel, tgt_in, tgt_out, mel_len, tgt_len) in enumerate(train_loader):

            mel     = mel.to(device, non_blocking=True)
            tgt_in  = tgt_in.to(device, non_blocking=True)
            tgt_out = tgt_out.to(device, non_blocking=True)
            mel_len = mel_len.to(device, non_blocking=True)

            # Forward
            with autocast("cuda", enabled=cfg.fp16):
                logits = model(mel, tgt_in, mel_len)
                loss   = compute_loss(logits, tgt_out, pad_id) / cfg.accum_steps

            #  Backward
            scaler.scale(loss).backward()
            epoch_loss += loss.item() * cfg.accum_steps

            # Update tous les accum_steps
            if (step + 1) % cfg.accum_steps == 0:
                # Dégradé le scaler avant le clip
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % cfg.log_every == 0:
                    avg  = epoch_loss / (step + 1)
                    lr   = scheduler.get_last_lr()[0]
                    vram = torch.cuda.memory_allocated() / 1e9 if device.type == "cuda" else 0
                    log.info(
                        f"Epoch {epoch+1} | Step {global_step} | "
                        f"Loss {avg:.4f} | LR {lr:.2e} | VRAM {vram:.1f}GB"
                    )
                    writer.add_scalar("train/loss", avg,  global_step)
                    writer.add_scalar("train/lr",   lr,   global_step)
                    writer.add_scalar("train/vram", vram, global_step)

        # Fin d'époque
        elapsed   = (time.time() - epoch_start) / 60
        avg_loss  = epoch_loss / max(len(train_loader), 1)
        log.info(f"\n{'='*55}")
        log.info(f"Époque {epoch+1}/{cfg.max_epochs} - {elapsed:.1f} min")
        log.info(f"  Loss train : {avg_loss:.4f}")

        # Évaluation
        if (epoch + 1) % cfg.eval_every == 0:
            log.info("  Évaluation...")
            val_loss, val_wer_score = evaluate(
                model, val_loader, tokenizer, device, cfg
            )
            log.info(f"  Loss val   : {val_loss:.4f}")
            log.info(f"  WER val    : {val_wer_score*100:.2f}%")

            writer.add_scalar("val/loss", val_loss,       epoch)
            writer.add_scalar("val/wer",  val_wer_score,  epoch)

            # Sauvegarder meilleur modèle
            if val_wer_score < best_wer:
                best_wer     = val_wer_score
                patience_ctr = 0
                best_path    = save_dir / "best_model.pt"
                torch.save({
                    "epoch":        epoch,
                    "model":        model.state_dict(),
                    "optimizer":    optimizer.state_dict(),
                    "best_wer":     best_wer,
                    "global_step":  global_step,
                    "model_config": model_cfg,
                    "train_config": cfg,
                }, best_path)
                log.info(f"  Meilleur modèle sauvegardé - WER={best_wer*100:.2f}%")
            else:
                patience_ctr += 1
                log.info(f"  Patience {patience_ctr}/{cfg.patience}")

        # Checkpoint régulier
        if (epoch + 1) % cfg.save_every == 0:
            torch.save({
                "epoch":        epoch,
                "model":        model.state_dict(),
                "optimizer":    optimizer.state_dict(),
                "best_wer":     best_wer,
                "global_step":  global_step,
                "model_config": model_cfg,
                "train_config": cfg,
            }, latest_ckpt)

        # Early Stopping
        if patience_ctr >= cfg.patience:
            log.info(f"\nEarly stopping - époque {epoch+1}")
            break

    writer.close()
    log.info(f"\nEntraînement terminé !")
    log.info(f"Meilleur WER : {best_wer*100:.2f}%")
    log.info(f"Modèle       : {save_dir / 'best_model.pt'}")


# MAIN
if __name__ == "__main__":
    # Vérifier que les fichiers requis existent
    cfg = TrainConfig()
    missing = []
    for path, name in [
        (cfg.train_manifest,  "train manifest"),
        (cfg.val_manifest,    "val manifest"),
        (cfg.tokenizer_path,  "tokenizer"),
    ]:
        if not Path(path).exists():
            missing.append(f"  ❌ {name} : {path}")

    if missing:
        print("Fichiers manquants :")
        for m in missing:
            print(m)
        print("\nLance d'abord :")
        print("  python 01_download_dataset.py")
        print("  python 02a_train_tokenizer.py")
        exit(1)

    train()