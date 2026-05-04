import json
import os
import sentencepiece as spm
from pathlib import Path

# CONFIGURATION
MANIFEST_DIR  = Path("./data/manifests")
TOKENIZER_DIR = Path("./data/tokenizer")
VOCAB_SIZE    = 1000    # 512–2000 est standard pour le français
MODEL_TYPE    = "bpe"   # "bpe" ou "unigram"
TEXT_FILE     = Path("./data/tokenizer/corpus.txt")

TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)


def extract_text_corpus():
    
    print("Extraction du corpus texte...")

    all_texts = []
    manifest_path = MANIFEST_DIR / "train.json"

    with open(manifest_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            text = entry['text'].lower().strip()
            # Nettoyage minimal
            text = text.replace('\n', ' ').strip()
            if text:
                all_texts.append(text)

    # Sauvegarder le corpus
    with open(TEXT_FILE, 'w', encoding='utf-8') as f:
        for text in all_texts:
            f.write(text + '\n')

    print(f"{len(all_texts)} phrases extraites → {TEXT_FILE}")
    return TEXT_FILE


def train_tokenizer():
    
    print(f"\n Entraînement tokenizer BPE (vocab={VOCAB_SIZE})...")

    model_prefix = str(TOKENIZER_DIR / "tokenizer_fr")

    spm.SentencePieceTrainer.train(
        input=str(TEXT_FILE),
        model_prefix=model_prefix,
        vocab_size=VOCAB_SIZE,
        model_type=MODEL_TYPE,
        character_coverage=1.0, # 1.0 (alphabet fr)
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece='<pad>',
        unk_piece='<unk>',
        bos_piece='<s>',
        eos_piece='</s>',
        user_defined_symbols=['<blank>'],   # Pour CTC
        num_threads=os.cpu_count(),
        input_sentence_size=5_000_000,
        shuffle_input_sentence=True,
    )

    print(f"Tokenizer sauvegardé : {model_prefix}.model")
    return f"{model_prefix}.model"


def test_tokenizer(model_path):
    
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)

    test_sentences = [
        "bonjour comment allez vous",
        "le chat est sur le tapis",
        "je vais prendre un café chaud",
    ]

    print("\nTest du tokenizer :")
    for sentence in test_sentences:
        tokens = sp.encode(sentence, out_type=str)
        ids    = sp.encode(sentence)
        print(f"   '{sentence}'")
        print(f"   → tokens : {tokens}")
        print(f"   → ids    : {ids}")
        print()

    print(f"   Taille vocabulaire : {sp.get_piece_size()}")
    print(f"   Token <blank> id   : {sp.piece_to_id('<blank>')}")


if __name__ == "__main__":
    print("="*60)
    print("ENTRAÎNEMENT TOKENIZER BPE")
    print("="*60)

    extract_text_corpus()
    model_path = train_tokenizer()
    test_tokenizer(model_path)

    print("\n Tokenizer prêt !")
    print(f"   → Modèle : {model_path}")
