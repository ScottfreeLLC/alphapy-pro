"""
Train a ByteLevelBPE tokenizer on the encoded price corpus.

Usage:
    python -m src.train_tokenizer --config config.yaml
"""

import argparse
from pathlib import Path

import yaml
from tokenizers import ByteLevelBPETokenizer


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer on price corpus")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    tok_config = config.get("tokenizer", {})

    corpus_path = Path("data/encoded/corpus.txt")
    if not corpus_path.exists():
        print("ERROR: corpus.txt not found. Run encode_corpus.py first.")
        return

    tokenizer_dir = Path("data/tokenizer")
    tokenizer_dir.mkdir(parents=True, exist_ok=True)

    vocab_size = tok_config.get("vocab_size", 2000)
    min_frequency = tok_config.get("min_frequency", 2)
    special_tokens = tok_config.get("special_tokens", [
        "<s>", "</s>", "<pad>", "<mask>", "<unk>",
        "boy", "eoy", "bom", "eom", "bod", "eod",
    ])

    print(f"Training ByteLevelBPE tokenizer (vocab_size={vocab_size})...")
    print(f"  Corpus: {corpus_path}")
    print(f"  Special tokens: {special_tokens}")

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[str(corpus_path)],
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
    )

    tokenizer.save_model(str(tokenizer_dir))

    # Test encoding
    with open(corpus_path) as f:
        sample = f.readline().strip()[:200]

    encoded = tokenizer.encode(sample)
    print(f"\nTokenizer trained: {len(tokenizer.get_vocab())} tokens")
    print(f"Sample: '{sample[:80]}...'")
    print(f"Encoded: {encoded.ids[:20]}...")
    print(f"Decoded: '{tokenizer.decode(encoded.ids[:20])}'")
    print(f"\nSaved to {tokenizer_dir}/")


if __name__ == "__main__":
    main()
