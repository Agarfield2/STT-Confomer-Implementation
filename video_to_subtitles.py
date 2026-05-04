"""
Pipeline complet : vidéo -> vidéo sous-titrée

Dépendances :
  pip install torch torchaudio sentencepiece soundfile tqdm
  + ffmpeg installé sur le système (https://ffmpeg.org/download.html)

Usage :
  python video_to_subtitles.py --video ma_video.mp4 --model models/best_model.pt
  python video_to_subtitles.py --video ma_video.mp4 --model models/best_model.pt --beam 3
  python video_to_subtitles.py --video ma_video.mp4 --model models/best_model.pt --srt_only
"""

import argparse
import subprocess
import sys
import os
import math
import tempfile
import re
from pathlib import Path
from dataclasses import dataclass

import torch
import torchaudio
import torchaudio.transforms as T
import sentencepiece as spm
import soundfile as sf
import numpy as np
from tqdm import tqdm

# Import au niveau module - indispensable pour que pickle trouve
# ModelConfig et TrainConfig lors du torch.load() du checkpoint.
try:
    from model import WhisperLike, ModelConfig
    from train  import TrainConfig
    from torch.serialization import add_safe_globals
    add_safe_globals([ModelConfig, TrainConfig])
except ImportError:
    # Les classes seront importées plus tard si le script est dans le bon dossier
    WhisperLike = None
    ModelConfig = None
    TrainConfig = None



# PARAMÈTRES AUDIO (identiques à dataset.py)

SAMPLE_RATE  = 16000
N_MELS       = 80
N_FFT        = 512
HOP_LENGTH   = 160     # 10 ms par frame
WIN_LENGTH   = 400     # 25 ms par fenêtre

# Taille d'un segment (en secondes)
SEGMENT_S    = 8.0    # < 30 s (limite du modèle)
OVERLAP_S    = 1.0     # chevauchement pour éviter les coupures de mots



# 1. EXTRACTION AUDIO DEPUIS LA VIDÉO

def extract_audio(video_path: str, tmp_dir: str) -> str:

    wav_path = os.path.join(tmp_dir, "audio_raw.wav")

    cmd = [
        "ffmpeg", "-y",           # -y = écraser sans demander
        "-i", video_path,
        "-vn",                    # pas de vidéo
        "-acodec", "pcm_s16le",   # WAV 16-bit PCM
        "-ar", str(SAMPLE_RATE),  # 16 000 Hz
        "-ac", "1",               # mono
        wav_path,
    ]

    print(f"\n[1/4] Extraction audio...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("Erreur ffmpeg :")
        print(result.stderr[-1000:])
        sys.exit(1)

    # Vérifier la durée
    duration = get_audio_duration_ffprobe(video_path)
    print(f"      Audio extrait : {duration:.1f}s -> {wav_path}")
    return wav_path, duration


def get_audio_duration_ffprobe(video_path: str) -> float:
    
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0



# 2. DÉCOUPAGE EN SEGMENTS

def split_audio_into_segments(wav_path: str, segment_s: float, overlap_s: float):

    data, sr = sf.read(wav_path, dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)

    # Rééchantillonnage de sécurité (ne devrait pas être nécessaire)
    if sr != SAMPLE_RATE:
        wt = torch.from_numpy(data).unsqueeze(0)
        import torchaudio.functional as AF
        wt = AF.resample(wt, sr, SAMPLE_RATE)
        data = wt.squeeze(0).numpy()

    total_samples  = len(data)
    seg_samples    = int(segment_s * SAMPLE_RATE)
    step_samples   = int((segment_s - overlap_s) * SAMPLE_RATE)

    segments = []
    start    = 0

    while start < total_samples:
        end   = min(start + seg_samples, total_samples)
        chunk = data[start:end]

        t_start = start / SAMPLE_RATE
        t_end   = end   / SAMPLE_RATE

        segments.append((chunk, t_start, t_end))
        start += step_samples

        if end == total_samples:
            break

    return segments


# 3. MEL SPECTROGRAM (identique à dataset.py)

_mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    win_length=WIN_LENGTH,
    n_mels=N_MELS,
    f_min=0.0,
    f_max=8000.0,
    power=2.0,
)

def audio_to_mel(waveform_np: np.ndarray) -> torch.Tensor:

    wt  = torch.from_numpy(waveform_np).unsqueeze(0)  # [1, T]
    mel = _mel_transform(wt)                            # [1, 80, T_frames]
    mel = torch.log(mel + 1e-9)
    mel = (mel - mel.mean()) / (mel.std() + 1e-8)
    mel = mel.squeeze(0).transpose(0, 1)               # [T_frames, 80]
    return mel.unsqueeze(0)                             # [1, T_frames, 80]



# 4. CHARGEMENT DU MODÈLE

def load_model(model_path: str, device: torch.device):

    print(f"\n[2/4] Chargement du modèle...")

    if WhisperLike is None:
        print("[ERREUR] model.py / train.py introuvables. Place ce script dans le même dossier.")
        sys.exit(1)

    ckpt      = torch.load(model_path, map_location=device, weights_only=False)
    model_cfg = ckpt["model_config"]
    train_cfg = ckpt.get("train_config", None)

    model = WhisperLike(model_cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Tokenizer
    tokenizer_path = (
        train_cfg.tokenizer_path
        if train_cfg
        else str(Path(model_path).parent.parent / "data" / "tokenizer" / "tokenizer_fr.model")
    )
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.load(tokenizer_path)

    params = model.count_parameters()
    print(f"      Modèle chargé : {params['total_M']}M paramètres")
    print(f"      Tokenizer     : {tokenizer.get_piece_size()} tokens")
    print(f"      Device        : {device}")

    return model, tokenizer



# 5. TRANSCRIPTION DES SEGMENTS

@dataclass
class Subtitle:
    start: float    # secondes
    end:   float    # secondes
    text:  str


def transcribe_segments(
    segments,
    model,
    tokenizer,
    device: torch.device,
    beam_size: int = 1,
    max_new_tokens: int = 200,
) -> list:

    print(f"\n[3/4] Transcription ({len(segments)} segments, beam={beam_size})...")

    eos_id  = tokenizer.piece_to_id("</s>")
    results = []

    with torch.no_grad():
        for chunk, t_start, t_end in tqdm(segments, desc="  Segments"):

            # Segment trop court -> ignorer
            if len(chunk) < SAMPLE_RATE * 0.3:
                continue

            # Mel spectrogram
            mel     = audio_to_mel(chunk).to(device)         # [1, T, 80]
            mel_len = torch.tensor([mel.shape[1]], device=device)

            # Inférence
            pred_ids = model.generate(
                mel,
                mel_len,
                max_new_tokens=max_new_tokens,
                beam_size=beam_size,
            )[0]

            text = tokenizer.decode(pred_ids).strip()

            # Ignorer les transcriptions vides ou parasites
            if not text or len(text) < 2:
                continue

            # Découper la transcription en sous-titres de ~7 mots max
            subs = split_text_into_subtitles(text, t_start, t_end)
            results.extend(subs)

    # Dédoublonner les chevauchements (dus à l'overlap)
    results = deduplicate_subtitles(results)

    print(f"      {len(results)} sous-titres générés")
    return results


def split_text_into_subtitles(
    text: str,
    t_start: float,
    t_end: float,
    max_words: int = 8,
) -> list:

    words    = text.split()
    duration = t_end - t_start

    if not words:
        return []

    # Calculer le temps par mot
    time_per_word = duration / max(len(words), 1)

    subtitles = []
    i = 0

    while i < len(words):
        chunk_words = words[i: i + max_words]
        chunk_start = t_start + i * time_per_word
        chunk_end   = t_start + (i + len(chunk_words)) * time_per_word
        chunk_end   = min(chunk_end, t_end)

        subtitles.append(Subtitle(
            start = round(chunk_start, 3),
            end   = round(chunk_end,   3),
            text  = " ".join(chunk_words),
        ))

        i += max_words

    return subtitles


def deduplicate_subtitles(subtitles: list, iou_threshold: float = 0.5) -> list:

    if not subtitles:
        return subtitles

    # Trier par temps de début
    subtitles.sort(key=lambda s: s.start)

    kept = [subtitles[0]]

    for sub in subtitles[1:]:
        last = kept[-1]

        # Chevauchement temporel
        overlap_start = max(sub.start, last.start)
        overlap_end   = min(sub.end,   last.end)
        overlap_dur   = max(0.0, overlap_end - overlap_start)

        dur_sub  = max(sub.end  - sub.start,  0.001)
        dur_last = max(last.end - last.start, 0.001)
        union    = dur_sub + dur_last - overlap_dur
        iou      = overlap_dur / max(union, 0.001)

        # Similarité textuelle basique (ratio de mots communs)
        words_sub  = set(sub.text.lower().split())
        words_last = set(last.text.lower().split())
        if words_sub and words_last:
            sim = len(words_sub & words_last) / max(len(words_sub | words_last), 1)
        else:
            sim = 0.0

        if iou > iou_threshold and sim > 0.8:
            continue  # doublon -> ignorer

        kept.append(sub)

    return kept



# 6. GÉNÉRATION DU FICHIER SRT

def seconds_to_srt_time(seconds: float) -> str:

    h   = int(seconds // 3600)
    m   = int((seconds % 3600) // 60)
    s   = int(seconds % 60)
    ms  = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(subtitles: list, output_path: str):

    with open(output_path, "w", encoding="utf-8") as f:
        for i, sub in enumerate(subtitles, 1):
            f.write(f"{i}\n")
            f.write(f"{seconds_to_srt_time(sub.start)} --> {seconds_to_srt_time(sub.end)}\n")
            f.write(f"{sub.text}\n\n")

    print(f"      SRT écrit : {output_path}")



# 7. BURN DES SOUS-TITRES DANS LA VIDÉO

def burn_subtitles(
    video_path: str,
    srt_path:   str,
    output_path: str,
    font_size: int = 24,
    font_color: str = "white",
    outline_color: str = "black",
):


    # Chemin absolu pour le SRT (requis par ffmpeg sur Windows)
    srt_abs = Path(srt_path).resolve()

    # Sur Windows, ffmpeg nécessite les backslashes doublés dans le filtre
    if sys.platform == "win32":
        srt_str = str(srt_abs).replace("\\", "/").replace(":", "\\:")
    else:
        srt_str = str(srt_abs).replace(":", "\\:")

    subtitle_filter = (
        f"subtitles='{srt_str}'"
        f":force_style='FontSize={font_size},"
        f"PrimaryColour=&H00FFFFFF,"   # blanc
        f"OutlineColour=&H00000000,"   # contour noir
        f"Outline=2,"
        f"Bold=1,"
        f"Alignment=2'"                # centré en bas
    )

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", subtitle_filter,
        "-c:v", "libx264",
        "-crf", "18",           # qualité proche de l'original
        "-preset", "fast",
        "-c:a", "copy",         # audio inchangé
        output_path,
    ]

    print(f"\n[4/4] Incrustation des sous-titres...")
    print(f"      Commande : {' '.join(cmd[:6])}...")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("Erreur ffmpeg burn :")
        print(result.stderr[-2000:])
        print("\nTentative avec soft subtitles...")
        burn_soft_subtitles(video_path, srt_path, output_path)
    else:
        print(f"      Vidéo sous-titrée : {output_path}")


def burn_soft_subtitles(video_path: str, srt_path: str, output_path: str):

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", srt_path,
        "-c:v", "copy",
        "-c:a", "copy",
        "-c:s", "mov_text",    # format sous-titres MP4
        "-map", "0:v",
        "-map", "0:a",
        "-map", "1:0",
        "-metadata:s:s:0", "language=fra",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print("Erreur soft subtitles :")
        print(result.stderr[-1000:])
        print(f"\nLe fichier SRT est disponible séparément : {srt_path}")
    else:
        print(f"      Soft subtitles ajoutés : {output_path}")


# 
# 8. PIPELINE PRINCIPAL
# 
def run_pipeline(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Vérifier les entrées
    if not Path(args.video).exists():
        print(f"Erreur : fichier vidéo introuvable : {args.video}")
        sys.exit(1)

    if not Path(args.model).exists():
        print(f"Erreur : checkpoint introuvable : {args.model}")
        sys.exit(1)

    # Préparer les chemins de sortie
    video_stem  = Path(args.video).stem
    output_dir  = Path(args.output_dir) if args.output_dir else Path(args.video).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    srt_path    = str(output_dir / f"{video_stem}.srt")
    output_vid  = str(output_dir / f"{video_stem}_subtitled{Path(args.video).suffix}")

    print("=" * 60)
    print("  PIPELINE VIDÉO -> VIDÉO SOUS-TITRÉE")
    print("=" * 60)
    print(f"  Vidéo   : {args.video}")
    print(f"  Modèle  : {args.model}")
    print(f"  Device  : {device}")
    print(f"  Beam    : {args.beam}")
    print(f"  Sortie  : {output_vid}")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmp_dir:

        # Étape 1 : extraire l'audio
        wav_path, duration = extract_audio(args.video, tmp_dir)

        # Étape 2 : découper en segments
        segments = split_audio_into_segments(wav_path, SEGMENT_S, OVERLAP_S)
        print(f"      {len(segments)} segments de {SEGMENT_S}s (overlap={OVERLAP_S}s)")

        # Étape 3 : charger le modèle
        model, tokenizer = load_model(args.model, device)

        # Étape 4 : transcrire
        subtitles = transcribe_segments(
            segments,
            model,
            tokenizer,
            device,
            beam_size=args.beam,
            max_new_tokens=args.max_tokens,
        )

        if not subtitles:
            print("\nAucun sous-titre généré. Vérifier la qualité audio et le modèle.")
            sys.exit(1)

        # Étape 5 : écrire le SRT
        write_srt(subtitles, srt_path)

        # Étape 6 : incruster ou fichier SRT uniquement
        if args.srt_only:
            print(f"\nMode --srt_only : vidéo non modifiée.")
            print(f"SRT disponible : {srt_path}")
        elif args.soft_subs:
            burn_soft_subtitles(args.video, srt_path, output_vid)
        else:
            burn_subtitles(args.video, srt_path, output_vid, font_size=args.font_size)

    print("\n" + "=" * 60)
    print("  TERMINÉ")
    print("=" * 60)
    if not args.srt_only:
        print(f"  Vidéo sous-titrée : {output_vid}")
    print(f"  Fichier SRT        : {srt_path}")
    print(f"  Sous-titres        : {len(subtitles)}")
    print("=" * 60)


# 
# 9. ARGS
# 
def parse_args():
    p = argparse.ArgumentParser(
        description="Pipeline vidéo -> vidéo sous-titrée avec ton modèle STT"
    )
    p.add_argument("--video",       required=True,  help="Chemin de la vidéo source (MP4, MKV, AVI...)")
    p.add_argument("--model",       required=True,  help="Chemin du checkpoint (best_model.pt)")
    p.add_argument("--output_dir",  default=None,   help="Dossier de sortie (défaut : même dossier que la vidéo)")
    p.add_argument("--beam",        type=int, default=1,   help="Taille du beam search (1=greedy, 3=recommandé)")
    p.add_argument("--max_tokens",  type=int, default=200, help="Tokens max par segment")
    p.add_argument("--font_size",   type=int, default=24,  help="Taille de la police des sous-titres")
    p.add_argument("--srt_only",    action="store_true",   help="Générer uniquement le .srt sans modifier la vidéo")
    p.add_argument("--soft_subs",   action="store_true",   help="Soft subtitles (piste séparée) au lieu de hardcoded")
    p.add_argument("--segment_s",   type=float, default=SEGMENT_S, help="Durée des segments audio (défaut: 28s)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Permettre de surcharger la taille des segments
    SEGMENT_S = args.segment_s

    run_pipeline(args)
