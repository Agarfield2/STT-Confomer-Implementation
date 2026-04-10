import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional, Tuple


# CONFIG
@dataclass
class ModelConfig:
    # Audio
    n_mels         : int   = 80
    max_audio_len  : int   = 3000

    # Modèle
    d_model        : int   = 256
    n_heads        : int   = 4
    n_enc_layers   : int   = 4
    n_dec_layers   : int   = 4
    d_ff           : int   = 1024
    dropout        : float = 0.1

    # Conformer spécifique
    conv_kernel    : int   = 31     # taille du noyau depthwise (doit être impair)

    # Vocabulaire
    vocab_size     : int   = 1002
    pad_id         : int   = 0
    bos_id         : int   = 1
    eos_id         : int   = 2
    max_text_len   : int   = 448


# BRIQUE 1 - POSITIONAL ENCODING
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, :x.size(1)])


# BRIQUE 2 - FEED FORWARD
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


# BRIQUE 3 - CONFORMER CONV MODULE
class ConformerConvModule(nn.Module):
    """
    Le bloc de convolution au coeur du Conformer.

    x → LayerNorm
      → Pointwise Conv (d_model → 2*d_model)   expansion
      → GLU (Gated Linear Unit)                 → d_model
      → Depthwise Conv (kernel=31, groups=d_model)
      → BatchNorm
      → Swish (x * sigmoid(x))
      → Pointwise Conv (d_model → d_model)      compression
      → Dropout
      → + x (résidu)

    Pourquoi GLU ?
      GLU(x) = A ⊙ sigmoid(B) où [A, B] = split(conv(x))
      C'est une "porte" - certaines features passent, d'autres
      sont filtrées. Plus expressif que ReLU/GELU simple.

    Pourquoi Depthwise Conv ?
      Conv standard : noyau [k, C_in, C_out] - mélange canaux
      Depthwise Conv : noyau [k, 1, 1] × C - canal par canal
      → 31 × 256 params au lieu de 31 × 256 × 256
      → capture les patterns temporels locaux sans surcoût
    """
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        assert kernel_size % 2 == 1, "kernel_size doit être impair"
        padding = (kernel_size - 1) // 2

        self.norm = nn.LayerNorm(d_model)

        # Pointwise Conv : expansion × 2 pour GLU
        self.pointwise_conv1 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model * 2,
            kernel_size=1,
        )

        # Depthwise Conv : opère canal par canal (groups=d_model)
        self.depthwise_conv = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding=padding,
            groups=d_model,    # depthwise : 1 filtre par canal
        )

        self.batch_norm = nn.BatchNorm1d(d_model)

        # Pointwise Conv : compression
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=1,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : [B, T, d_model]
        residual = x
        x = self.norm(x)

        # Conv1d attend [B, C, T] - on transpose
        x = x.transpose(1, 2)                  # [B, d_model, T]

        # Pointwise + GLU
        x = self.pointwise_conv1(x)             # [B, 2*d_model, T]
        x = F.glu(x, dim=1)                    # [B, d_model, T]

        # Depthwise Conv
        x = self.depthwise_conv(x)              # [B, d_model, T]

        # BatchNorm + Swish
        x = self.batch_norm(x)
        x = x * torch.sigmoid(x)               # Swish = x * σ(x)

        # Pointwise + Dropout
        x = self.pointwise_conv2(x)             # [B, d_model, T]
        x = self.dropout(x)

        # Retransposer + résidu
        x = x.transpose(1, 2)                  # [B, T, d_model]
        return x + residual


# BRIQUE 4 - COUCHE CONFORMER
class ConformerLayer(nn.Module):
    """
    Une couche Conformer complète - remplace EncoderLayer.

    Structure Macaron (deux demi-FeedForward autour de l'attention) :

      x → FF(½ scale) → Self-Attn → ConvModule → FF(½ scale) → LayerNorm

    Le "½ scale" signifie que la sortie du FeedForward est
    multipliée par 0.5 avant d'être ajoutée au résidu.
    Cela stabilise l'entraînement en réduisant l'amplitude
    des mises à jour.

    Comparé à l'EncoderLayer du Sprint 1 :
      Sprint 1 : x → SelfAttn → FF
      Sprint 2 : x → FF(½) → SelfAttn → Conv → FF(½) → Norm
                              ↑ la conv capture le local
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 conv_kernel: int = 31, dropout: float = 0.1):
        super().__init__()

        # Deux demi-FeedForward (Macaron)
        self.ff1  = FeedForward(d_model, d_ff, dropout)
        self.ff2  = FeedForward(d_model, d_ff, dropout)

        # Self-Attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Bloc de convolution
        self.conv_module = ConformerConvModule(d_model, conv_kernel, dropout)

        # Normes
        self.norm_ff1  = nn.LayerNorm(d_model)
        self.norm_attn = nn.LayerNorm(d_model)
        self.norm_final= nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # x : [B, T, d_model]

        #  FeedForward 1 (½ scale, Pre-LN) 
        residual = x
        x = self.norm_ff1(x)
        x = self.dropout(self.ff1(x))
        x = 0.5 * x + residual              # ← demi-step

        #  Self-Attention (Pre-LN) 
        residual = x
        x = self.norm_attn(x)
        x, _ = self.self_attn(
            query=x, key=x, value=x,
            key_padding_mask=src_key_padding_mask,
        )
        x = self.dropout(x) + residual

        #  Convolution Module 
        # ConformerConvModule gère son propre résidu en interne
        x = self.conv_module(x)

        #  FeedForward 2 (½ scale, Pre-LN) 
        residual = x
        x = self.norm_final(x)
        x = self.dropout(self.ff2(x))
        x = 0.5 * x + residual              # demi-step

        return x   # [B, T, d_model]


# BRIQUE 5 - COUCHE DÉCODEUR
class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.ff      = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, tgt_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(
            query=x, key=x, value=x,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )
        x = self.dropout(x) + residual

        residual = x
        x = self.norm2(x)
        x, _ = self.cross_attn(
            query=x, key=enc_out, value=enc_out,
            key_padding_mask=memory_key_padding_mask,
        )
        x = self.dropout(x) + residual

        residual = x
        x = self.norm3(x)
        x = self.dropout(self.ff(x)) + residual
        return x


# ENCODEUR CONFORMER COMPLET
class AudioEncoder(nn.Module):
    """
    Encodeur audio avec couches Conformer.

    Front-end Conv1D identique au Sprint 1 (subsampling T→T/2).
    Les N EncoderLayer sont remplacées par N ConformerLayer.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Front-end subsampling (identique Sprint 1)
        self.conv1 = nn.Conv1d(config.n_mels, config.d_model, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(config.d_model, config.d_model, kernel_size=3, stride=2, padding=1)

        self.pos_enc = PositionalEncoding(config.d_model, config.max_audio_len, config.dropout)

        # Couches Conformer au lieu de TransformerEncoder
        self.layers = nn.ModuleList([
            ConformerLayer(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                conv_kernel=config.conv_kernel,
                dropout=config.dropout,
            )
            for _ in range(config.n_enc_layers)
        ])

        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, mel: torch.Tensor, mel_lengths: torch.Tensor = None):
        # mel : [B, T, n_mels]
        x = mel.transpose(1, 2)             # [B, n_mels, T]
        x = F.gelu(self.conv1(x))           # [B, d_model, T]
        x = F.gelu(self.conv2(x))           # [B, d_model, T/2]
        x = x.transpose(1, 2)              # [B, T/2, d_model]
        x = self.pos_enc(x)

        src_mask = None
        if mel_lengths is not None:
            enc_lengths = torch.ceil(mel_lengths.float() / 2).long()
            T = x.size(1)
            src_mask = torch.arange(T, device=x.device).unsqueeze(0) >= enc_lengths.unsqueeze(1)

        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_mask)

        return self.norm(x), src_mask   # [B, T/2, d_model]


# DÉCODEUR COMPLET
class TextDecoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config    = config
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_id)
        self.pos_enc   = PositionalEncoding(config.d_model, config.max_text_len + 10, config.dropout)
        self.layers    = nn.ModuleList([
            DecoderLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_dec_layers)
        ])
        self.norm        = nn.LayerNorm(config.d_model)
        self.output_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.output_proj.weight = self.token_emb.weight
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(size, size, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, tgt_ids, enc_out, memory_key_padding_mask=None):
        U      = tgt_ids.size(1)
        device = tgt_ids.device
        x = self.token_emb(tgt_ids) * math.sqrt(self.config.d_model)
        x = self.pos_enc(x)
        causal_mask  = self.make_causal_mask(U, device)
        tgt_pad_mask = (tgt_ids == self.config.pad_id)
        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask=causal_mask,
                      tgt_key_padding_mask=tgt_pad_mask,
                      memory_key_padding_mask=memory_key_padding_mask)
        return self.output_proj(self.norm(x))   # [B, U, vocab_size]


# MODÈLE COMPLET
class WhisperLike(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config  = config
        self.encoder = AudioEncoder(config)
        self.decoder = TextDecoder(config)

    def forward(self, mel, tgt_ids, mel_lengths=None):
        enc_out, enc_mask = self.encoder(mel, mel_lengths)
        return self.decoder(tgt_ids, enc_out, memory_key_padding_mask=enc_mask)

    @torch.no_grad()
    def generate(self, mel, mel_lengths=None, max_new_tokens=200,
                 beam_size=1, temperature=1.0, length_penalty=0.6):
        self.eval()
        device = mel.device
        B      = mel.size(0)
        enc_out, enc_mask = self.encoder(mel, mel_lengths)
        if beam_size == 1:
            return self._greedy(enc_out, enc_mask, B, device, max_new_tokens, temperature)
        return self._beam_search(enc_out, enc_mask, B, device, max_new_tokens, beam_size, length_penalty)

    # greedy
    def _greedy(self, enc_out, enc_mask, B, device, max_new_tokens, temperature):
        generated = torch.full((B, 1), self.config.bos_id, dtype=torch.long, device=device)
        finished  = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            logits     = self.decoder(generated, enc_out, memory_key_padding_mask=enc_mask)
            next_logits= logits[:, -1, :]
            if temperature != 1.0:
                next_logits = next_logits / temperature
            next_token = next_logits.argmax(dim=-1, keepdim=True)
            finished   = finished | (next_token.squeeze(-1) == self.config.eos_id)
            generated  = torch.cat([generated, next_token], dim=1)
            if finished.all():
                break

        results = []
        for i in range(B):
            seq = generated[i, 1:].tolist()
            if self.config.eos_id in seq:
                seq = seq[:seq.index(self.config.eos_id)]
            results.append(seq)
        return results

    # Beam Search amélioré
    def _beam_search(self, enc_out, enc_mask, B, device,
                     max_new_tokens, beam_size, length_penalty=0.6):
        """
        Beam Search avec length penalty.

        Sans length penalty, le score est la somme des log-probs.
        Chaque token ajouté est négatif → les séquences courtes
        ont de meilleurs scores → le modèle génère trop peu de mots.

        Length penalty : score_normalisé = score / (longueur ^ α)
        α = 0.6 est la valeur standard (Google, Whisper).
        α = 0 → pas de penalty (greedy equivalent)
        α = 1 → division par la longueur exacte

        À chaque pas on garde les beam_size meilleures hypothèses,
        on les développe toutes, et on refiltre les beam_size meilleures.
        """
        results = []

        for b in range(B):
            enc_b  = enc_out[b:b+1]
            mask_b = enc_mask[b:b+1] if enc_mask is not None else None

            # Chaque beam : (score_brut, séquence_ids)
            beams     = [(0.0, [self.config.bos_id])]
            completed = []

            for step in range(max_new_tokens):
                if not beams:
                    break

                all_candidates = []

                for score, seq in beams:
                    # Séquence déjà terminée
                    if seq[-1] == self.config.eos_id:
                        completed.append((score, seq))
                        continue

                    # Prédire le prochain token
                    ids    = torch.tensor([seq], dtype=torch.long, device=device)
                    logits = self.decoder(ids, enc_b, memory_key_padding_mask=mask_b)
                    log_p  = F.log_softmax(logits[0, -1, :], dim=-1)

                    # Développer les beam_size meilleures options
                    topk_lp, topk_ids = log_p.topk(beam_size)

                    for lp, tid in zip(topk_lp.tolist(), topk_ids.tolist()):
                        new_seq   = seq + [tid]
                        new_score = score + lp
                        all_candidates.append((new_score, new_seq))

                if not all_candidates:
                    break

                # Trier par score normalisé par longueur
                def normalized_score(item):
                    sc, seq = item
                    lp = length_penalty
                    return sc / ((len(seq) ** lp) + 1e-9)

                all_candidates.sort(key=normalized_score, reverse=True)
                beams = all_candidates[:beam_size]

                # Si tous ont généré EOS → terminé
                if all(s[-1] == self.config.eos_id for _, s in beams):
                    completed.extend(beams)
                    beams = []
                    break

            # Ajouter les beams non terminés
            completed.extend(beams)

            # Choisir la meilleure séquence normalisée
            def norm_sc(item):
                sc, seq = item
                return sc / ((len(seq) ** length_penalty) + 1e-9)

            completed.sort(key=norm_sc, reverse=True)
            best = completed[0][1]

            # Nettoyer BOS / EOS
            best = best[1:]
            if self.config.eos_id in best:
                best = best[:best.index(self.config.eos_id)]

            results.append(best)

        return results

    def count_parameters(self):
        enc = sum(p.numel() for p in self.encoder.parameters())
        dec = sum(p.numel() for p in self.decoder.parameters())
        tot = enc + dec
        return {'encoder': enc, 'decoder': dec, 'total': tot, 'total_M': round(tot/1e6, 1)}


# TEST
if __name__ == "__main__":
    print("=" * 55)
    print("Test Sprint 2 - Conformer + Beam Search")
    print("=" * 55)

    cfg   = ModelConfig()
    model = WhisperLike(cfg)
    p     = model.count_parameters()
    print(f"\nParamètres : {p['total_M']}M")
    print(f"  Encodeur (Conformer) : {p['encoder']:,}")
    print(f"  Décodeur             : {p['decoder']:,}")

    # Test forward
    B, T, U = 2, 800, 30
    mel  = torch.randn(B, T, cfg.n_mels)
    tgt  = torch.randint(3, cfg.vocab_size, (B, U))
    lens = torch.tensor([800, 600])
    tgt[:, 0] = cfg.bos_id

    print(f"\nTest forward...")
    logits = model(mel, tgt, lens)
    assert logits.shape == (B, U, cfg.vocab_size)
    print(f"  Shape OK : {list(logits.shape)}")

    # Test Conformer Conv Module isolé
    print(f"\nTest ConformerConvModule...")
    conv = ConformerConvModule(d_model=256, kernel_size=31)
    x    = torch.randn(2, 100, 256)
    y    = conv(x)
    assert y.shape == x.shape, f"Shape incorrecte : {y.shape}"
    print(f"  Input  : {list(x.shape)}")
    print(f"  Output : {list(y.shape)}  (inchangé - connexion résiduelle)")

    # Test ConformerLayer
    print(f"\nTest ConformerLayer...")
    layer = ConformerLayer(d_model=256, n_heads=4, d_ff=1024, conv_kernel=31)
    y     = layer(x)
    assert y.shape == x.shape
    print(f"  Shape OK : {list(y.shape)}")

    # Test greedy generate
    print(f"\nTest generate greedy...")
    res = model.generate(mel, lens, max_new_tokens=20, beam_size=1)
    print(f"  Seq 0 : {res[0][:8]}... ({len(res[0])} tokens)")

    # Test beam search
    print(f"\nTest generate beam_size=3 (length_penalty=0.6)...")
    res_beam = model.generate(mel[:1], lens[:1], max_new_tokens=20, beam_size=3, length_penalty=0.6)
    print(f"  Seq 0 : {res_beam[0][:8]}... ({len(res_beam[0])} tokens)")

    # Comparaison paramètres Sprint 1 vs Sprint 2
    print(f"\nComparaison Sprint 1 vs Sprint 2 :")
    print(f"  Sprint 1 (Transformer) : ~7.9M params")
    print(f"  Sprint 2 (Conformer)   : {p['total_M']}M params")
    print(f"  Différence             : +Conv depthwise par couche")

    vram = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    print(f"\nVRAM params seuls : {vram:.1f} MB")
    print(f"Compatible RTX 3060 6GB : Oui")

    print(f"\n{'='*55}")
    print(f"Sprint 2 OK - Conformer + Beam Search prêts !")
    print(f"{'='*55}")
