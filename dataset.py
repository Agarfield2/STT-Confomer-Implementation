import json
import math
import random
from pathlib import Path
from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import numpy as np
import sentencepiece as spm
from torch.utils.data import Dataset, DataLoader


# CONFIG AUDIO
SAMPLE_RATE  = 16000
N_MELS       = 80
N_FFT        = 512
HOP_LENGTH   = 160    # 10ms par frame
WIN_LENGTH   = 400    # 25ms par fenêtre
MAX_AUDIO_S  = 30.0   # secondes max (= 3000 frames à 10ms)
MIN_AUDIO_S  = 0.5    # secondes min


# DATASET PRINCIPAL
class SpeechDataset(Dataset):


    def __init__(
        self,
        manifest_path: str,
        tokenizer_path: str,
        augment: bool = False,
        max_audio_s: float = MAX_AUDIO_S,
        min_audio_s: float = MIN_AUDIO_S,
        max_tokens: int = 440,
        sort_by_duration: bool = True,   # curriculum learning
    ):
        self.augment      = augment
        self.max_audio_s  = max_audio_s
        self.min_audio_s  = min_audio_s
        self.max_tokens   = max_tokens

        # Charger le tokenizer
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(tokenizer_path)

        # IDs spéciaux
        self.bos_id = self.sp.piece_to_id('<s>')
        self.eos_id = self.sp.piece_to_id('</s>')
        self.pad_id = self.sp.piece_to_id('<pad>')

        # Charger le manifest
        self.samples = []
        with open(manifest_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)

                # Filtrer par durée
                dur = entry.get('duration', 0)
                if dur < min_audio_s or dur > max_audio_s:
                    continue

                # Filtrer les transcriptions vides
                text = entry.get('text', '').strip()
                if not text:
                    continue

                self.samples.append(entry)

        # Curriculum learning - trier par durée croissante
        # Le modèle voit d'abord les exemples courts (plus faciles)
        if sort_by_duration:
            self.samples.sort(key=lambda x: x.get('duration', 0))

        print(f"   Dataset chargé : {len(self.samples)} samples")
        print(f"   BOS={self.bos_id}, EOS={self.eos_id}, PAD={self.pad_id}")

        # Transforms audio
        self.mel_transform = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            win_length=WIN_LENGTH,
            n_mels=N_MELS,
            f_min=0.0,
            f_max=8000.0,
            power=2.0,
        )

        # SpecAugment - masquage aléatoire de fréquences et de temps
        # Améliore la robustesse du modèle au bruit
        if augment:
            self.freq_mask = T.FrequencyMasking(freq_mask_param=27)
            self.time_mask = T.TimeMasking(time_mask_param=100, p=0.05)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]

        # Charger et préparer l'audio
        mel = self._load_audio(sample['audio_filepath'])

        # Tokeniser le texte 
        text    = sample['text'].lower().strip()
        tok_ids = self.sp.encode(text)

        # Tronquer si trop long
        tok_ids = tok_ids[:self.max_tokens]

        # Teacher forcing - décalage d'un pas
        # tgt_in  = [<bos>, tok0, tok1, ..., tokN]
        # tgt_out = [tok0,  tok1, ..., tokN, <eos>]
        tgt_in  = torch.tensor([self.bos_id] + tok_ids,          dtype=torch.long)
        tgt_out = torch.tensor(tok_ids + [self.eos_id],          dtype=torch.long)

        return mel, tgt_in, tgt_out

    def _load_audio(self, audio_path: str) -> torch.Tensor:
        

        # Charger le WAV avec soundfile (pas de dépendance torchcodec)
        import soundfile as sf
        import numpy as np
        data, sr = sf.read(audio_path, dtype='float32', always_2d=False)
        # data : [T] mono ou [T, C] multi-canal

        if data.ndim == 2:
            data = data.mean(axis=1)   # → mono [T]

        waveform = torch.from_numpy(data).unsqueeze(0)   # [1, T]

        # Rééchantillonner si nécessaire
        if sr != SAMPLE_RATE:
            import torchaudio.functional as AF
            waveform = AF.resample(waveform, sr, SAMPLE_RATE)

        # Mel spectrogram [1, n_mels, T]
        mel = self.mel_transform(waveform)

        # Log compression - réduit la plage dynamique
        # +1e-9 évite log(0)
        mel = torch.log(mel + 1e-9)

        # SpecAugment (entraînement seulement)
        if self.augment:
            mel = self.freq_mask(mel)
            mel = self.time_mask(mel)

        # Normalisation par utterance
        # Soustrait la moyenne, divise par l'écart-type
        # Chaque clip a une moyenne=0 et écart-type=1
        mel = (mel - mel.mean()) / (mel.std() + 1e-8)

        # [1, n_mels, T] → [T, n_mels]
        mel = mel.squeeze(0).transpose(0, 1)

        return mel   # [T, n_mels]


# COLLATE - PADDING DYNAMIQUE
def collate_fn(batch, pad_id: int = 0):

    mels, tgt_ins, tgt_outs = zip(*batch)

    # Longueurs réelles
    mel_lengths = torch.tensor([m.shape[0] for m in mels],    dtype=torch.long)
    tgt_lengths = torch.tensor([t.shape[0] for t in tgt_ins], dtype=torch.long)

    T_max = mel_lengths.max().item()
    U_max = tgt_lengths.max().item()
    B     = len(mels)
    n_mel = mels[0].shape[1]

    # Padding mel avec zéros (silence en log-mel)
    mel_padded = torch.zeros(B, T_max, n_mel)
    for i, m in enumerate(mels):
        mel_padded[i, :m.shape[0], :] = m

    # Padding tokens avec pad_id
    tgt_in_padded  = torch.full((B, U_max), pad_id, dtype=torch.long)
    tgt_out_padded = torch.full((B, U_max), pad_id, dtype=torch.long)
    for i, (ti, to) in enumerate(zip(tgt_ins, tgt_outs)):
        tgt_in_padded[i,  :ti.shape[0]] = ti
        tgt_out_padded[i, :to.shape[0]] = to

    return mel_padded, tgt_in_padded, tgt_out_padded, mel_lengths, tgt_lengths


def make_collate_fn(pad_id: int):
    
    def _collate(batch):
        return collate_fn(batch, pad_id=pad_id)
    return _collate


# CRÉATION DES DATALOADERS
def get_dataloaders(
    train_manifest: str,
    val_manifest: str,
    tokenizer_path: str,
    batch_size: int = 8,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader]:
    

    print("Chargement train dataset...")
    train_ds = SpeechDataset(
        manifest_path=train_manifest,
        tokenizer_path=tokenizer_path,
        augment=True,
        sort_by_duration=True,
    )

    print("Chargement validation dataset...")
    val_ds = SpeechDataset(
        manifest_path=val_manifest,
        tokenizer_path=tokenizer_path,
        augment=False,
        sort_by_duration=False,
    )

    # Récupérer le pad_id depuis le tokenizer
    sp      = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)
    pad_id  = sp.piece_to_id('<pad>')
    collate = make_collate_fn(pad_id)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,          # déjà trié par curriculum
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size * 2,   # pas de gradient → batch plus grand
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate,
    )

    return train_loader, val_loader


# TEST
if __name__ == "__main__":
    import sys
    print("=" * 55)
    print("Test du Dataset Encoder-Decoder")
    print("=" * 55)

    # Chemins
    MANIFEST  = "G:/Armand/New-Whisper/data/manifests/train.json"
    TOKENIZER = "G:/Armand/New-Whisper/data/tokenizer/tokenizer_fr.model"

    # Vérifier que les fichiers existent
    if not Path(MANIFEST).exists():
        print(f"\n Manifest introuvable : {MANIFEST}")
        print("   Lance d'abord : python 01_download_dataset.py")
        sys.exit(1)

    if not Path(TOKENIZER).exists():
        print(f"\n Tokenizer introuvable : {TOKENIZER}")
        print("   Lance d'abord : python 02a_train_tokenizer.py")
        sys.exit(1)

    # Créer le dataset
    print("\nChargement dataset...")
    ds = SpeechDataset(
        manifest_path=MANIFEST,
        tokenizer_path=TOKENIZER,
        augment=False,
    )

    print(f"\nTaille dataset : {len(ds)}")

    # Tester un sample
    mel, tgt_in, tgt_out = ds[0]
    print(f"\nSample 0 :")
    print(f"  mel     : {list(mel.shape)}  (T frames × {N_MELS} bandes)")
    print(f"  tgt_in  : {list(tgt_in.shape)}  = {tgt_in[:5].tolist()}...")
    print(f"  tgt_out : {list(tgt_out.shape)}  = {tgt_out[:5].tolist()}...")

    # Vérifier le teacher forcing
    # tgt_in[1:] doit être égal à tgt_out[:-1]
    assert torch.equal(tgt_in[1:], tgt_out[:-1]), "Erreur teacher forcing !"
    print(f"  Teacher forcing OK - tgt_in[1:] == tgt_out[:-1]")

    # Décoder pour vérifier
    sp   = spm.SentencePieceProcessor()
    sp.load(TOKENIZER)
    text_in  = sp.decode(tgt_in[1:].tolist())
    text_out = sp.decode(tgt_out[:-1].tolist())
    print(f"\n  Texte décodé tgt_in  : '{text_in}'")
    print(f"  Texte décodé tgt_out : '{text_out}'")
    print(f"  Référence manifest   : '{ds.samples[0]['text']}'")

    # Tester le collate
    print(f"\nTest collate (batch=4) ...")
    sp2    = spm.SentencePieceProcessor()
    sp2.load(TOKENIZER)
    pad_id = sp2.piece_to_id('<pad>')

    loader = DataLoader(
        ds,
        batch_size=4,
        shuffle=False,
        collate_fn=make_collate_fn(pad_id)
    )
    mel_b, tgt_in_b, tgt_out_b, mel_len, tgt_len = next(iter(loader))

    print(f"  mel_padded    : {list(mel_b.shape)}")
    print(f"  tgt_in_padded : {list(tgt_in_b.shape)}")
    print(f"  tgt_out_padded: {list(tgt_out_b.shape)}")
    print(f"  mel_lengths   : {mel_len.tolist()}")
    print(f"  tgt_lengths   : {tgt_len.tolist()}")

    # Vérifier que BOS est bien en première position
    assert (tgt_in_b[:, 0] == sp2.piece_to_id('<s>')).all(), "BOS manquant !"
    print(f"  BOS en position 0 : OK")

    # Vérifier que EOS est bien à la fin de chaque séquence
    for i, tl in enumerate(tgt_len):
        last_token = tgt_out_b[i, tl - 1].item()
        assert last_token == sp2.piece_to_id('</s>'), f"EOS manquant pour sample {i}"
    print(f"  EOS en dernière position : OK")

    print(f"\n{'='*55}")
    print(f"Dataset OK !")
    print(f"Prochaine étape : python 02d_train.py")
    print(f"{'='*55}")
