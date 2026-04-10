import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# CONFIG
@dataclass
class ModelConfig:
    # Audio
    n_mels        : int   = 80       # bandes Mel
    max_audio_len : int   = 3000     # frames max (30 sec à 10ms/frame)

    # Modèle
    d_model       : int   = 256      # dimension interne - réduit pour 3080
    n_heads       : int   = 4        # têtes d'attention (d_model / n_heads = 64)
    n_enc_layers  : int   = 4        # couches encodeur
    n_dec_layers  : int   = 4        # couches décodeur
    d_ff          : int   = 1024     # dimension Feed Forward (4 × d_model)
    dropout       : float = 0.1

    # Vocabulaire
    vocab_size    : int   = 1002     # 1000 BPE + <pad>=0 + <bos>=1 + <eos>=2
    pad_id        : int   = 0
    bos_id        : int   = 1
    eos_id        : int   = 2
    max_text_len  : int   = 448      # tokens max en sortie



# BRIQUE 1 - POSITIONAL ENCODING

class PositionalEncoding(nn.Module):
    """
    Encodage positionnel sinusoïdal.

    Le Transformer n'a pas de notion d'ordre - toutes les positions
    sont traitées en parallèle. On ajoute donc un vecteur unique
    par position, calculé avec des sinus/cosinus à différentes fréquences.

    sin(pos / 10000^(2i/d)) pour les dimensions paires
    cos(pos / 10000^(2i/d)) pour les dimensions impaires

    Résultat : deux positions proches ont des encodages similaires,
    deux positions éloignées ont des encodages très différents.
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # Calcul des encodages positionnels - fait une seule fois
        pe  = torch.zeros(max_len, d_model)                          # [max_len, d_model]
        pos = torch.arange(0, max_len).unsqueeze(1).float()          # [max_len, 1]
        div = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )                                                             # [d_model/2]

        pe[:, 0::2] = torch.sin(pos * div)   # dimensions paires
        pe[:, 1::2] = torch.cos(pos * div)   # dimensions impaires

        # register_buffer = sauvegardé dans state_dict mais pas entraîné
        self.register_buffer('pe', pe.unsqueeze(0))                  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : [B, T, d_model]
        # on ajoute l'encodage positionnel aux T premières positions
        return self.dropout(x + self.pe[:, :x.size(1)])



# BRIQUE 2 - FEED FORWARD NETWORK

class FeedForward(nn.Module):
    """
    Réseau feed-forward appliqué position par position.

    Chaque token passe indépendamment dans :
      Linear(d_model → d_ff) → GELU → Dropout → Linear(d_ff → d_model)

    Le d_ff est typiquement 4× d_model - c'est une expansion puis
    une compression. C'est ici que le modèle stocke ses "connaissances"
    sur les patterns qu'il a appris.

    GELU (Gaussian Error Linear Unit) remplace ReLU dans les Transformers
    modernes - transition plus douce autour de 0.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : [B, T, d_model]
        return self.linear2(
            self.dropout(
                F.gelu(self.linear1(x))
            )
        )



# BRIQUE 3 - COUCHE ENCODEUR

class EncoderLayer(nn.Module):
    """
    Une couche d'encodeur = Self-Attention + Feed Forward.

    Self-Attention : chaque frame audio regarde toutes les autres frames
    pour comprendre le contexte - "ce son ressemble à un 'b' surtout
    parce qu'il est suivi d'une voyelle".

    Architecture Pre-LN (Layer Norm avant l'attention) :
    Plus stable à l'entraînement que Post-LN.

    x → LayerNorm → SelfAttention → + x (résidu)
      → LayerNorm → FeedForward   → + x (résidu)

    La connexion résiduelle (+ x) permet aux gradients de circuler
    directement sans traverser les couches - essentiel pour entraîner
    des réseaux profonds.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True    # [B, T, d_model] au lieu de [T, B, d_model]
        )
        self.ff      = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        # x : [B, T, d_model]

        # Self-Attention (Pre-LN) ---
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(
            query=x, key=x, value=x,
            key_padding_mask=src_key_padding_mask  # ignore les positions padding
        )
        x = self.dropout(x) + residual

        # Feed Forward (Pre-LN) ---
        residual = x
        x = self.norm2(x)
        x = self.dropout(self.ff(x)) + residual

        return x   # [B, T, d_model]



# BRIQUE 4 - COUCHE DÉCODEUR

class DecoderLayer(nn.Module):
    """
    Une couche de décodeur = Masked Self-Attention + Cross-Attention + FF.

    Masked Self-Attention : le token à la position U ne peut voir
    que les tokens 0..U-1, jamais le futur. Sinon le modèle
    "tricherait" pendant l'entraînement en lisant la réponse.
    Le masque causal est un masque triangulaire supérieur.

    Cross-Attention : les tokens décodés "écoutent" la sortie
    de l'encodeur. C'est ici que le lien audio ↔ texte se crée.
    Query = tokens décodés, Key/Value = sortie encodeur.

    x → LayerNorm → MaskedSelfAttn → + x
      → LayerNorm → CrossAttn(enc)  → + x
      → LayerNorm → FeedForward     → + x
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Self-attention masquée (ne voit pas le futur)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        # Cross-attention (écoute l'encodeur)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )

        self.ff      = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        # x       : [B, U, d_model] - tokens décodés jusqu'ici
        # enc_out : [B, T, d_model] - sortie encodeur

        #  Masked Self-Attention
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attn(
            query=x, key=x, value=x,
            attn_mask=tgt_mask,                    # masque causal triangulaire
            key_padding_mask=tgt_key_padding_mask  # padding des tokens cibles
        )
        x = self.dropout(x) + residual

        # Cross-Attention
        residual = x
        x = self.norm2(x)
        x, _ = self.cross_attn(
            query=x,                               # ce que le décodeur cherche
            key=enc_out,                           # ce que l'encodeur a trouvé
            value=enc_out,
            key_padding_mask=memory_key_padding_mask  # padding de l'audio
        )
        x = self.dropout(x) + residual

        # Feed Forward
        residual = x
        x = self.norm3(x)
        x = self.dropout(self.ff(x)) + residual

        return x   # [B, U, d_model]



# ENCODEUR COMPLET

class AudioEncoder(nn.Module):
    """
    Encodeur audio complet.

    Conv1D front-end : réduit la séquence T → T/2 et extrait
    des features locales (un peu comme un CNN sur spectrogramme).
    Deux convolutions de stride 1 avec GELU.

    Puis N couches EncoderLayer empilées.
    LayerNorm final pour stabiliser la sortie.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Front-end Conv1D - opère sur la dimension temporelle
        # Conv1d attend [B, C, T] donc on transpose avant/après
        self.conv1 = nn.Conv1d(
            in_channels=config.n_mels,
            out_channels=config.d_model,
            kernel_size=3,
            padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=config.d_model,
            out_channels=config.d_model,
            kernel_size=3,
            stride=2,        # ← divise T par 2
            padding=1
        )

        self.pos_enc = PositionalEncoding(
            config.d_model,
            max_len=config.max_audio_len,
            dropout=config.dropout
        )

        self.layers = nn.ModuleList([
            EncoderLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_enc_layers)
        ])

        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        mel: torch.Tensor,
        mel_lengths: torch.Tensor = None
    ):
        # mel : [B, T, n_mels]

        # Conv1D attend [B, C, T]
        x = mel.transpose(1, 2)            # [B, n_mels, T]
        x = F.gelu(self.conv1(x))          # [B, d_model, T]
        x = F.gelu(self.conv2(x))          # [B, d_model, T/2]
        x = x.transpose(1, 2)             # [B, T/2, d_model]

        # Encodage positionnel
        x = self.pos_enc(x)

        # Masque de padding (positions audio vides après le vrai signal)
        src_mask = None
        if mel_lengths is not None:
            # Les longueurs sont réduites de moitié par la conv stride=2
            enc_lengths = torch.ceil(mel_lengths.float() / 2).long()
            T = x.size(1)
            # True = position à ignorer (padding)
            src_mask = torch.arange(T, device=x.device).unsqueeze(0) >= enc_lengths.unsqueeze(1)

        # Passer dans les N couches
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_mask)

        x = self.norm(x)   # [B, T/2, d_model]
        return x, src_mask



# DÉCODEUR COMPLET

class TextDecoder(nn.Module):
    """
    Décodeur texte complet.

    Embedding : convertit les ids de tokens en vecteurs d_model.
    Positional Encoding : ajoute l'information de position.
    N couches DecoderLayer empilées.
    Projection finale vers le vocabulaire.

    Le masque causal est généré automatiquement pour bloquer
    le futur lors de la self-attention.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.d_model,
            padding_idx=config.pad_id
        )

        self.pos_enc = PositionalEncoding(
            config.d_model,
            max_len=config.max_text_len + 10,
            dropout=config.dropout
        )

        self.layers = nn.ModuleList([
            DecoderLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_dec_layers)
        ])

        self.norm    = nn.LayerNorm(config.d_model)

        # Projection finale : d_model → vocab_size
        self.output_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Partage des poids embedding ↔ projection (technique Whisper)
        # Réduit les paramètres et améliore la généralisation
        self.output_proj.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        """Initialisation Xavier pour stabiliser l'entraînement."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        """
        Masque causal triangulaire.
        True = position bloquée (ne peut pas voir ce token).

        Position 0 voit : [0]
        Position 1 voit : [0, 1]
        Position 2 voit : [0, 1, 2]
        ...

        Représenté comme matrice booléenne upper-triangular :
        [[F, T, T],
         [F, F, T],
         [F, F, F]]
        """
        mask = torch.triu(
            torch.ones(size, size, dtype=torch.bool, device=device),
            diagonal=1   # au-dessus de la diagonale = futur = bloqué
        )
        return mask

    def forward(
        self,
        tgt_ids: torch.Tensor,
        enc_out: torch.Tensor,
        memory_key_padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        # tgt_ids : [B, U] - tokens cibles (décalés d'un pas)
        # enc_out : [B, T, d_model] - sortie encodeur

        U      = tgt_ids.size(1)
        device = tgt_ids.device

        # Embedding + positional encoding
        x = self.token_emb(tgt_ids) * math.sqrt(self.config.d_model)
        x = self.pos_enc(x)                       # [B, U, d_model]

        # Masque causal
        causal_mask = self.make_causal_mask(U, device)

        # Masque de padding sur les tokens cibles
        tgt_pad_mask = (tgt_ids == self.config.pad_id)   # True = padding

        # Passer dans les N couches
        for layer in self.layers:
            x = layer(
                x, enc_out,
                tgt_mask=causal_mask,
                tgt_key_padding_mask=tgt_pad_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )

        x = self.norm(x)
        logits = self.output_proj(x)   # [B, U, vocab_size]
        return logits



# MODÈLE COMPLET

class WhisperLike(nn.Module):
    """
    Modèle complet Encoder-Decoder.

    Deux modes :
      - forward()  : entraînement (teacher forcing)
      - generate() : inférence (génération autoregressive)
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config  = config
        self.encoder = AudioEncoder(config)
        self.decoder = TextDecoder(config)

    def forward(
        self,
        mel: torch.Tensor,
        tgt_ids: torch.Tensor,
        mel_lengths: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Passe forward pour l'entraînement.

        Teacher forcing : on donne les vrais tokens en entrée du décodeur,
        même si le modèle s'est trompé au pas précédent.
        Plus rapide à entraîner, mais crée un décalage avec l'inférence.

        tgt_ids en entrée  = [<bos>, tok1, tok2, ..., tokN]
        logits en sortie   = [tok1,  tok2, ..., tokN, <eos>]
        → décalé d'un pas : on prédit le prochain token à chaque position
        """
        # Encoder l'audio
        enc_out, enc_mask = self.encoder(mel, mel_lengths)

        # Décoder les tokens (décalés : entrée = tgt[:-1], cible = tgt[1:])
        logits = self.decoder(tgt_ids, enc_out, memory_key_padding_mask=enc_mask)

        return logits   # [B, U, vocab_size]

    @torch.no_grad()
    def generate(
        self,
        mel: torch.Tensor,
        mel_lengths: torch.Tensor = None,
        max_new_tokens: int = 200,
        beam_size: int = 1,      # 1 = greedy, >1 = beam search
        temperature: float = 1.0
    ) -> list:
        """
        Génération autoregressive token par token.

        Mode greedy (beam_size=1) :
          À chaque pas, prend le token le plus probable.
          Rapide, moins précis.

        Mode beam search (beam_size>1) :
          Garde les N meilleures hypothèses en parallèle.
          Plus lent, plus précis.
        """
        self.eval()
        device = mel.device
        B      = mel.size(0)

        # Encoder l'audio une seule fois
        enc_out, enc_mask = self.encoder(mel, mel_lengths)

        if beam_size == 1:
            return self._greedy_decode(enc_out, enc_mask, B, device, max_new_tokens, temperature)
        else:
            return self._beam_search(enc_out, enc_mask, B, device, max_new_tokens, beam_size)

    def _greedy_decode(self, enc_out, enc_mask, B, device, max_new_tokens, temperature):
        """Décodage greedy - prend le token le plus probable à chaque pas."""
        # Initialiser avec le token BOS
        generated = torch.full((B, 1), self.config.bos_id, dtype=torch.long, device=device)
        finished  = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            # Calculer les logits pour la séquence générée jusqu'ici
            logits = self.decoder(generated, enc_out, memory_key_padding_mask=enc_mask)

            # Prendre le dernier token uniquement
            next_logits = logits[:, -1, :]   # [B, vocab_size]

            # Appliquer la température (>1 = plus aléatoire, <1 = plus concentré)
            if temperature != 1.0:
                next_logits = next_logits / temperature

            # Greedy : argmax
            next_token = next_logits.argmax(dim=-1, keepdim=True)   # [B, 1]

            # Marquer les séquences terminées
            finished = finished | (next_token.squeeze(-1) == self.config.eos_id)

            # Ajouter le nouveau token
            generated = torch.cat([generated, next_token], dim=1)

            # Arrêter si toutes les séquences sont terminées
            if finished.all():
                break

        # Convertir en listes, retirer le BOS initial
        results = []
        for i in range(B):
            seq = generated[i, 1:].tolist()   # retirer BOS
            # Couper à EOS si présent
            if self.config.eos_id in seq:
                seq = seq[:seq.index(self.config.eos_id)]
            results.append(seq)

        return results

    def _beam_search(self, enc_out, enc_mask, B, device, max_new_tokens, beam_size):
        """
        Beam search - garde les beam_size meilleures hypothèses.

        À chaque pas :
          1. Pour chaque hypothèse active, calculer prob de chaque token
          2. Garder les beam_size meilleures combinaisons (hypothèse + nouveau token)
          3. Répéter jusqu'à EOS ou max_new_tokens

        Score = somme des log-probabilités (pour éviter l'underflow)
        """
        results = []

        for b in range(B):
            # Extraire la représentation encodeur pour cet item
            enc_b    = enc_out[b:b+1]                      # [1, T, d_model]
            mask_b   = enc_mask[b:b+1] if enc_mask is not None else None

            # Initialiser les beams : (score, séquence)
            beams     = [(0.0, [self.config.bos_id])]
            completed = []

            for _ in range(max_new_tokens):
                all_candidates = []

                for score, seq in beams:
                    if seq[-1] == self.config.eos_id:
                        completed.append((score, seq))
                        continue

                    # Prédire le prochain token
                    ids    = torch.tensor([seq], dtype=torch.long, device=device)
                    logits = self.decoder(ids, enc_b, memory_key_padding_mask=mask_b)
                    log_probs = F.log_softmax(logits[0, -1, :], dim=-1)   # [vocab]

                    # Top-k candidats pour éviter l'explosion combinatoire
                    topk_logprobs, topk_ids = log_probs.topk(beam_size)

                    for lp, tid in zip(topk_logprobs.tolist(), topk_ids.tolist()):
                        all_candidates.append((score + lp, seq + [tid]))

                if not all_candidates:
                    break

                # Garder les beam_size meilleurs
                all_candidates.sort(key=lambda x: x[0], reverse=True)
                beams = all_candidates[:beam_size]

                # Arrêter si tous les beams ont généré EOS
                if all(s[-1] == self.config.eos_id for _, s in beams):
                    for s, seq in beams:
                        completed.append((s, seq))
                    break

            # Prendre la meilleure séquence complète
            completed += beams
            completed.sort(key=lambda x: x[0], reverse=True)
            best_seq = completed[0][1]

            # Nettoyer BOS/EOS
            best_seq = best_seq[1:]   # retirer BOS
            if self.config.eos_id in best_seq:
                best_seq = best_seq[:best_seq.index(self.config.eos_id)]

            results.append(best_seq)

        return results

    def count_parameters(self) -> dict:
        """Compte les paramètres par composant."""
        enc_params = sum(p.numel() for p in self.encoder.parameters())
        dec_params = sum(p.numel() for p in self.decoder.parameters())
        total      = enc_params + dec_params
        return {
            'encoder': enc_params,
            'decoder': dec_params,
            'total':   total,
            'total_M': round(total / 1e6, 1)
        }



# TEST RAPIDE

if __name__ == "__main__":
    print("="*55)
    print("Test de l'architecture WhisperLike")
    print("="*55)

    cfg   = ModelConfig()
    model = WhisperLike(cfg)
    params = model.count_parameters()

    print(f"\nConfig :")
    print(f"  d_model      = {cfg.d_model}")
    print(f"  n_heads      = {cfg.n_heads}")
    print(f"  enc_layers   = {cfg.n_enc_layers}")
    print(f"  dec_layers   = {cfg.n_dec_layers}")
    print(f"  d_ff         = {cfg.d_ff}")
    print(f"  vocab_size   = {cfg.vocab_size}")

    print(f"\nParamètres :")
    print(f"  Encodeur  : {params['encoder']:,}")
    print(f"  Décodeur  : {params['decoder']:,}")
    print(f"  Total     : {params['total']:,}  ({params['total_M']}M)")

    # Test forward (entraînement)
    B, T, U = 2, 800, 30
    mel     = torch.randn(B, T, cfg.n_mels)
    tgt     = torch.randint(3, cfg.vocab_size, (B, U))
    tgt[:, 0] = cfg.bos_id
    lengths = torch.tensor([800, 600])

    print(f"\nTest forward (batch={B}, T={T}, U={U}) ...")
    logits = model(mel, tgt, lengths)
    print(f"  Entrée  mel  : {list(mel.shape)}")
    print(f"  Entrée  tgt  : {list(tgt.shape)}")
    print(f"  Sortie logits: {list(logits.shape)}")
    assert logits.shape == (B, U, cfg.vocab_size), "Shape incorrecte !"
    print(f"  Shape OK")

    # Test generate (inférence greedy)
    print(f"\nTest generate greedy (batch={B}) ...")
    results = model.generate(mel, lengths, max_new_tokens=20)
    for i, r in enumerate(results):
        print(f"  Seq {i} : {r[:10]}... ({len(r)} tokens)")

    # Test generate (beam search)
    print(f"\nTest generate beam_size=3 (batch=1) ...")
    results_beam = model.generate(mel[:1], lengths[:1], max_new_tokens=15, beam_size=3)
    print(f"  Seq 0 : {results_beam[0][:10]}... ({len(results_beam[0])} tokens)")

    # VRAM estimée
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    print(f"\nVRAM estimée (paramètres seuls) : {param_bytes / 1e6:.1f} MB")
    print(f"  Avec activations (batch=8, FP16) : ~2-4 GB")
    print(f"  Compatible RTX 3080 (10 GB) : Oui")

    print(f"\n{'='*55}")
    print(f"Architecture prête !")
    print(f"Prochaine étape : python 02c_dataset.py")
    print(f"{'='*55}")
