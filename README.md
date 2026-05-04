# STT Pipeline - Français / Québécois
### Modèle de transcription automatique from scratch sur RTX 3080

---

## Structure du projet

```
stt_project/
├── scripts/
│   ├── download_dataset.py    ← Téléchargement Common Voice FR
│   ├── train_tokenizer.py    ← Tokenizer BPE (à faire avant l'entraînement)
│   ├── train.py        ← Architecture + boucle d'entraînement
│   └── evaluate_wer.py        ← Évaluation WER FR vs QC
│   └── video_to_subtitles.py        ← Pipline complet pour des vidéos
├── data/
│   ├── common_voice_fr/          ← Audios WAV 16kHz (auto-créé)
│   ├── manifests/                ← Fichiers JSON manifest (auto-créé)
│   └── tokenizer/                ← Modèle BPE (auto-créé)
├── models/                       ← Checkpoints (auto-créé)
├── logs/                         ← Logs + TensorBoard (auto-créé)
├── evaluation/                   ← Rapport WER HTML (auto-créé)
└── requirements.txt
```

---

## Installation

```bash
# 1. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 2. Installer PyTorch avec CUDA
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Se connecter à Hugging Face (requis pour Common Voice)
huggingface-cli login
# → Aller sur https://huggingface.co/settings/tokens
# → Créer un token avec accès read
# → Accepter les conditions sur :
#   https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0
```

---

## Lancement étape par étape

### ÉTAPE 1 - Téléchargement du dataset (~20 GB, ~30-60 min)
```bash
python scripts/01_download_dataset.py
```
**Ce qui se passe :**
- Télécharge Common Voice 17.0 FR depuis Hugging Face
- Convertit tous les audios en WAV 16kHz mono
- Filtre les clips entre 1 et 15 secondes
- Limite le train à 150h (configurable)
- Extrait les clips québécois dans des manifests séparés
- Génère des fichiers `train.json`, `validation.json`, `test.json`

**Résultat attendu :**
```
 train      : ~50 000 clips | ~150h
 validation : ~15 000 clips | ~45h
 test       : ~15 000 clips | ~45h
```

---

### ÉTAPE 2a - Tokenizer BPE (~5 min)
```bash
python scripts/02a_train_tokenizer.py
```
**Ce qui se passe :**
- Extrait tout le texte des transcriptions
- Entraîne un tokenizer BPE avec vocabulaire de 1000 tokens
- Adapté au français avec caractères accentués

**Résultat attendu :**
```
 data/tokenizer/tokenizer_fr.model
 data/tokenizer/tokenizer_fr.vocab
```

---

### ÉTAPE 2b - Entraînement du modèle (3-5 jours sur RTX 3080)
```bash
python scripts/02b_train_model.py
```
**Architecture :**
```
CNN Subsampling (x4) → Positional Encoding → Transformer Encoder (6 layers) → CTC Loss
```

**Optimisations RTX 3080 :**
- FP16 Mixed Precision → ~6 GB VRAM utilisés
- Gradient Accumulation x8 → batch effectif de 64
- Curriculum Learning → courts audios d'abord

**Suivre l'entraînement en temps réel :**
```bash
# Terminal 2 - TensorBoard
tensorboard --logdir ./logs/tensorboard
# → Ouvrir http://localhost:6006

# Terminal 3 - GPU monitoring
watch -n 2 nvidia-smi
```

**Reprendre un entraînement interrompu :**
```bash
# Le script reprend automatiquement depuis le dernier checkpoint
python scripts/02b_train_model.py
```

---

### ÉTAPE 3 - Évaluation WER (~15 min)
```bash
python scripts/03_evaluate_wer.py
```
**Ce qui se passe :**
- Charge le meilleur modèle sauvegardé
- Calcule le WER sur le test set français standard
- Calcule le WER sur les clips québécois uniquement
- Analyse les patterns d'erreurs (substitutions, suppressions, insertions)
- Identifie les mots québécois mal reconnus
- Génère un rapport HTML complet avec graphiques

**Résultat attendu :**
```
evaluation/
├── rapport_wer.html     ← Rapport complet (ouvrir dans navigateur)
├── wer_comparison.png   ← Graphique WER
└── summary.json         ← Métriques JSON
```

---

## WER attendus selon les données

| Données QC dans le train | WER FR Standard | WER Québécois |
|---|---|---|
| <5% (dataset FR pur) | ~15% | ~35-45% |
| ~15% (Common Voice FR) | ~15% | ~25-35% |
| 50%+ (données QC ajoutées) | ~15-18% | ~15-20% |

---
## Pipeline complet : vidéo -> vidéo sous-titrée

#### Usage :
  python video_to_subtitles.py --video ma_video.mp4 --model models/best_model.pt
  python video_to_subtitles.py --video ma_video.mp4 --model models/best_model.pt --beam 3
  python video_to_subtitles.py --video ma_video.mp4 --model models/best_model.pt --srt_only
---

## Suivant la carte graphique :

Si tu manques de VRAM :
```python
# Dans Config() de 02b_train_model.py :
cfg.d_model           = 128   # Réduire de 256 à 128
cfg.num_encoder_layers = 4    # Réduire de 6 à 4
cfg.batch_size        = 4     # Réduire de 8 à 4
cfg.accum_steps       = 16    # Augmenter pour compenser
```

Si tu veux aller plus vite (moins bonne qualité) :
```python
cfg.max_audio_len = 10.0   # Max 10 secondes au lieu de 15
cfg.n_mels        = 40     # 40 mel bins au lieu de 80
```

---

## Problèmes courants

**CUDA out of memory**
```bash
# Vider le cache GPU
python -c "import torch; torch.cuda.empty_cache()"
# Puis réduire batch_size dans la config
```

