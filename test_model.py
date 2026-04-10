import torch
import sentencepiece as spm

from model import WhisperLike
from dataset import get_dataloaders
from train import TrainConfig  # ton fichier renommé
from torch.serialization import add_safe_globals
from model import ModelConfig

# Autoriser le chargement du checkpoint
add_safe_globals([ModelConfig, TrainConfig])

MODEL_PATH = "G:/Armand/New-Whisper/models/best_model.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Load checkpoint
ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)

model_cfg = ckpt["model_config"]
cfg       = ckpt["train_config"]

model = WhisperLike(model_cfg).to(DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()

# Tokenizer
tokenizer = spm.SentencePieceProcessor()
tokenizer.load(cfg.tokenizer_path)

pad_id = tokenizer.piece_to_id("<pad>")
eos_id = tokenizer.piece_to_id("</s>")

# DataLoader EXACT du training
_, val_loader = get_dataloaders(
    train_manifest=cfg.train_manifest,
    val_manifest=cfg.val_manifest,
    tokenizer_path=cfg.tokenizer_path,
    batch_size=1,
    num_workers=0,
)

print("\n===== TEST RAPIDE SUR 3 PHRASES DU VAL SET =====\n")

# Test sur 3 samples
with torch.no_grad():
    for i, (mel, tgt_in, tgt_out, mel_len, tgt_len) in enumerate(val_loader):
        if i == 3:
            break

        mel = mel.to(DEVICE)
        mel_len = mel_len.to(DEVICE)

        # Génération autoregressive
        pred_ids = model.generate(
            mel,
            mel_len,
            max_new_tokens=100,
            beam_size=1
        )[0]

        # Référence
        tgt_ids = tgt_out[0][:tgt_len[0]].tolist()
        if eos_id in tgt_ids:
            tgt_ids = tgt_ids[:tgt_ids.index(eos_id)]

        ref = tokenizer.decode(tgt_ids).lower().strip()
        hyp = tokenizer.decode(pred_ids).lower().strip()

        print("=" * 60)
        print("REF :", ref)
        print("HYP :", hyp)
        print("")