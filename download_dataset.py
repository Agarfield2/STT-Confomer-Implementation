import json
import tarfile
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import librosa
import soundfile as sf
import numpy as np


# CONFIG

DATA_DIR     = Path("G:/Armand/New-Whisper/data/common_voice")
OUTPUT_DIR   = Path("G:/Armand/New-Whisper/data/common_voice_wav")
MANIFEST_DIR = Path("G:/Armand/New-Whisper/data/manifests")

TARGET_SR    = 16000
MIN_DURATION = 1.0
MAX_DURATION = 15.0
TARGET_HOURS = 150

# Chemins exacts détectés
AUDIO_DIR    = DATA_DIR / "audio" / "fr"
TSV_DIR      = DATA_DIR / "transcript" / "fr"

QC_KEYWORDS  = ['canadian', 'québécois', 'quebecois', 'quebec', 'canada', 'montreal']

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)



# 1. EXTRACTION DES .TAR

def extract_all_tars():
    """Extrait tous les .tar dans leur dossier respectif."""
    tar_files = list(AUDIO_DIR.rglob("*.tar"))

    if not tar_files:
        print("   Aucun .tar trouvé - déjà extraits ?")
        return

    print(f"\n {len(tar_files)} fichier(s) .tar à extraire...")
    print("    Cela peut prendre 20-40 minutes pour les 15 fichiers train...\n")

    for tar_path in tar_files:
        # Dossier de destination = même dossier que le .tar
        extract_dir = tar_path.parent / tar_path.stem
        if extract_dir.exists() and any(extract_dir.rglob("*.mp3")):
            print(f"     Déjà extrait : {tar_path.name}")
            continue

        extract_dir.mkdir(exist_ok=True)
        print(f"    Extraction : {tar_path.name} → {extract_dir.name}/")
        try:
            with tarfile.open(tar_path, 'r') as tar:
                tar.extractall(path=extract_dir)
            # Compter les MP3 extraits
            mp3_count = len(list(extract_dir.rglob("*.mp3")))
            print(f"       {mp3_count} fichiers MP3 extraits")
        except Exception as e:
            print(f"       Erreur : {e}")



# 2. INDEX DES FICHIERS MP3

def build_audio_index():
    """
    Indexe tous les MP3 trouvés : {stem: chemin_complet}

    Structure réelle détectée :
      audio/fr/train/fr_train_0/fr_train_0/common_voice_fr_XXXXX.mp3
      audio/fr/other/fr_other_0/fr_other_0/common_voice_fr_XXXXX.mp3
    Le TSV contient juste : common_voice_fr_XXXXX.mp3
    → on indexe par stem (nom sans extension)
    """
    print("\n Indexation des fichiers MP3...")

    index = {}
    # Chercher dans tout le dossier common_voice récursivement
    search_root = DATA_DIR
    mp3_files   = list(search_root.rglob("*.mp3"))

    for mp3 in mp3_files:
        stem = mp3.stem   # ex: "common_voice_fr_18045394"
        if stem not in index:
            index[stem] = mp3

    print(f"    {len(index)} fichiers MP3 indexés")

    if index:
        # Afficher un exemple pour vérifier
        sample_key = next(iter(index))
        print(f"   Exemple : {sample_key} → {index[sample_key]}")
    else:
        print("    Aucun MP3 trouvé !")
        print(f"   Vérifie que les .tar sont bien extraits dans : {AUDIO_DIR}")

    return index



# 3. CONVERSION MP3 → WAV 16kHz

def convert_to_wav(mp3_path, wav_path):
    """
    Convertit un MP3 en WAV 16kHz mono normalisé.
    Utilise librosa qui gère les MP3 nativement sur Windows.
    Retourne la durée en secondes.
    """
    import librosa
    import soundfile as sf

    # librosa charge directement en float32 mono à n'importe quel sr
    waveform, sr = librosa.load(str(mp3_path), sr=TARGET_SR, mono=True)
    # waveform : numpy array float32 [T]

    # Normalisation
    max_val = abs(waveform).max()
    if max_val > 0:
        waveform = waveform / max_val

    duration = len(waveform) / TARGET_SR

    # Sauvegarder en WAV
    sf.write(str(wav_path), waveform, TARGET_SR, subtype='PCM_16')
    return duration



# 4. TRAITEMENT D'UN SPLIT

def process_split(tsv_path, split_name, audio_index, max_hours=None):
    """Lit le TSV, convertit les audios, génère le manifest."""
    print(f"\n  Traitement '{split_name}'...")

    # Lire le TSV
    df = pd.read_csv(tsv_path, sep='\t', low_memory=False)
    print(f"   {len(df)} entrées dans {tsv_path.name}")

    wav_dir = OUTPUT_DIR / split_name
    wav_dir.mkdir(parents=True, exist_ok=True)

    manifest          = []
    total_duration    = 0.0
    max_secs          = (max_hours * 3600) if max_hours else float('inf')
    skip_missing      = 0
    skip_duration     = 0
    skip_error        = 0

    for idx, row in enumerate(tqdm(df.itertuples(), total=len(df), desc=f"  {split_name}")):

        if total_duration >= max_secs:
            print(f"\n    Quota {max_hours}h atteint ({total_duration/3600:.1f}h)")
            break

        try:
            # Nom du fichier audio depuis la colonne 'path'
            raw_path = getattr(row, 'path', None)
            if not raw_path:
                skip_missing += 1
                continue

            # Chercher dans l'index par stem (sans extension)
            stem     = Path(str(raw_path)).stem
            mp3_path = audio_index.get(stem)

            if mp3_path is None:
                skip_missing += 1
                continue

            # Transcription
            sentence = str(getattr(row, 'sentence', '') or '').strip()
            if not sentence:
                skip_missing += 1
                continue

            # Chemin WAV de sortie
            wav_path = wav_dir / f"{split_name}_{idx:07d}.wav"

            # Conversion
            duration = convert_to_wav(mp3_path, wav_path)

            # Filtre durée
            if duration < MIN_DURATION or duration > MAX_DURATION:
                wav_path.unlink(missing_ok=True)
                skip_duration += 1
                continue

            # Accent québécois
            accent = str(getattr(row, 'accents', '') or '').strip()
            is_qc  = any(kw in accent.lower() for kw in QC_KEYWORDS)

            manifest.append({
                "audio_filepath": str(wav_path.resolve()),
                "text":           sentence,
                "duration":       round(duration, 3),
                "accent":         accent,
                "is_qc":          is_qc,
                "age":            str(getattr(row, 'age',    '') or ''),
                "gender":         str(getattr(row, 'gender', '') or ''),
            })
            total_duration += duration

        except Exception as e:
            skip_error += 1
            if skip_error <= 3:
                print(f"   Warning sample {skip_error}: {type(e).__name__}: {e}")
            continue

    # Stats
    qc = sum(1 for e in manifest if e['is_qc'])
    print(f"\n    Résultat '{split_name}' :")
    print(f"       Clips convertis  : {len(manifest)}")
    print(f"        Fichiers manquants: {skip_missing}")
    print(f"        Hors durée       : {skip_duration}")
    print(f"       Erreurs          : {skip_error}")
    print(f"        Durée totale     : {total_duration/3600:.2f}h")
    print(f"       Clips QC         : {qc} ({qc/max(len(manifest),1)*100:.1f}%)")

    return manifest



# 5. SAUVEGARDE MANIFESTS

def save_manifest(manifest, name):
    path = MANIFEST_DIR / f"{name}.json"
    with open(path, 'w', encoding='utf-8') as f:
        for entry in manifest:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    print(f"    {path.name} - {len(manifest)} entrées")

    # Manifest QC séparé
    qc_entries = [e for e in manifest if e['is_qc']]
    if qc_entries:
        qc_path = MANIFEST_DIR / f"{name}_qc.json"
        with open(qc_path, 'w', encoding='utf-8') as f:
            for entry in qc_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
        print(f"    {qc_path.name} - {len(qc_entries)} entrées QC")



# MAIN

if __name__ == "__main__":
    print("=" * 60)
    print("  PRÉPARATION COMMON VOICE FR - STT Pipeline")
    print("=" * 60)
    print(f"   AUDIO    : {AUDIO_DIR}")
    print(f"   TSV      : {TSV_DIR}")
    print(f"   WAV OUT  : {OUTPUT_DIR}")
    print(f"   MANIFESTS: {MANIFEST_DIR}")

    # 1. Extraire les .tar
    extract_all_tars()

    # 2. Indexer les MP3 
    audio_index = build_audio_index()
    if not audio_index:
        print("\n Arrêt - aucun MP3 disponible.")
        exit(1)

    # 3. Traiter chaque split
    all_manifests = {}

    # TRAIN - utilise validated.tsv (meilleure qualité)
    validated_tsv = TSV_DIR / "validated.tsv"
    train_tsv     = TSV_DIR / "train.tsv"
    tsv_to_use    = validated_tsv if validated_tsv.exists() else train_tsv

    if tsv_to_use.exists():
        m = process_split(tsv_to_use, "train", audio_index, max_hours=TARGET_HOURS)
        save_manifest(m, "train")
        all_manifests["train"] = m
    else:
        print(f" TSV train introuvable : {tsv_to_use}")

    # VALIDATION
    dev_tsv = TSV_DIR / "dev.tsv"
    if dev_tsv.exists():
        m = process_split(dev_tsv, "validation", audio_index)
        save_manifest(m, "validation")
        all_manifests["validation"] = m

    # TEST
    test_tsv = TSV_DIR / "test.tsv"
    if test_tsv.exists():
        m = process_split(test_tsv, "test", audio_index)
        save_manifest(m, "test")
        all_manifests["test"] = m

    # 4. Résumé
    print("\n" + "=" * 60)
    print(" RÉSUMÉ FINAL")
    print("=" * 60)
    total_clips = 0
    total_hours = 0
    for split, manifest in all_manifests.items():
        h  = sum(e['duration'] for e in manifest) / 3600
        qc = sum(1 for e in manifest if e['is_qc'])
        print(f"  {split.upper():12s}: {len(manifest):6d} clips | {h:6.1f}h | QC: {qc}")
        total_clips += len(manifest)
        total_hours += h
    print(f"  {'TOTAL':12s}: {total_clips:6d} clips | {total_hours:6.1f}h")
    print("=" * 60)
    print(" Dataset prêt !")
    print(f"   Manifests → {MANIFEST_DIR}")
    print("\n Prochaine étape :")
    print("   python 02a_train_tokenizer.py")