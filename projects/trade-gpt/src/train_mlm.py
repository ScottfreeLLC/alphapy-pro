"""
Train a RoBERTa-style masked language model on encoded price sequences.

Uses HuggingFace Transformers with DataCollatorForLanguageModeling.

Usage:
    python -m src.train_mlm --config config.yaml [--resume CHECKPOINT]
"""

import argparse
from pathlib import Path

import yaml
from datasets import Dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import (
    DataCollatorForLanguageModeling,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizerFast,
    Trainer,
    TrainingArguments,
)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_tokenizer(tokenizer_dir: str) -> RobertaTokenizerFast:
    """Load the trained BPE tokenizer as a RobertaTokenizerFast."""
    tokenizer = RobertaTokenizerFast(
        vocab_file=f"{tokenizer_dir}/vocab.json",
        merges_file=f"{tokenizer_dir}/merges.txt",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        mask_token="<mask>",
        unk_token="<unk>",
    )
    return tokenizer


def load_corpus(corpus_path: str, tokenizer, max_length: int) -> Dataset:
    """Load and tokenize the corpus into a HuggingFace Dataset."""
    with open(corpus_path) as f:
        lines = [line.strip() for line in f if line.strip()]

    # Chunk long sequences into max_length windows
    all_input_ids = []
    for line in lines:
        encoding = tokenizer(line, truncation=False, add_special_tokens=False)
        ids = encoding["input_ids"]
        # Sliding window with stride
        stride = max_length // 2
        for i in range(0, max(1, len(ids) - max_length + 1), stride):
            chunk = ids[i : i + max_length]
            if len(chunk) >= 32:  # Skip very short chunks
                # Pad if needed
                chunk = chunk + [tokenizer.pad_token_id] * (max_length - len(chunk))
                all_input_ids.append(chunk)

    print(f"Created {len(all_input_ids)} training sequences of length {max_length}")

    dataset = Dataset.from_dict({
        "input_ids": all_input_ids,
        "attention_mask": [
            [1 if t != tokenizer.pad_token_id else 0 for t in ids]
            for ids in all_input_ids
        ],
    })
    return dataset


def main():
    parser = argparse.ArgumentParser(description="Train RoBERTa MLM on price sequences")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--resume", default=None, help="Resume from checkpoint directory")
    args = parser.parse_args()

    config = load_config(args.config)
    mlm_config = config["mlm"]

    tokenizer_dir = Path("data/tokenizer")
    corpus_path = Path("data/encoded/corpus.txt")
    output_dir = Path("models") / mlm_config["model_name"]

    if not corpus_path.exists():
        print("ERROR: corpus.txt not found. Run encode_corpus.py first.")
        return
    if not (tokenizer_dir / "vocab.json").exists():
        print("ERROR: Tokenizer not found. Run train_tokenizer.py first.")
        return

    # Load tokenizer
    tokenizer = load_tokenizer(str(tokenizer_dir))
    vocab_size = len(tokenizer)
    print(f"Tokenizer loaded: {vocab_size} tokens")

    # Build model
    model_config = RobertaConfig(
        vocab_size=vocab_size,
        hidden_size=mlm_config["hidden_dim"],
        num_hidden_layers=mlm_config["num_layers"],
        num_attention_heads=mlm_config["num_heads"],
        intermediate_size=mlm_config["intermediate_dim"],
        max_position_embeddings=mlm_config["max_length"] + 2,
        type_vocab_size=1,
    )

    if args.resume:
        model = RobertaForMaskedLM.from_pretrained(args.resume)
        print(f"Resumed from {args.resume}")
    else:
        model = RobertaForMaskedLM(model_config)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"Model created: {n_params:,} parameters")

    # Load dataset
    dataset = load_corpus(str(corpus_path), tokenizer, mlm_config["max_length"])

    # Split train/eval
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]

    # Data collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=mlm_config["mlm_probability"],
    )

    # Training
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=mlm_config["epochs"],
        per_device_train_batch_size=mlm_config["batch_size"],
        per_device_eval_batch_size=mlm_config["batch_size"],
        learning_rate=mlm_config["learning_rate"],
        warmup_steps=mlm_config["warmup_steps"],
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="none",
        fp16=False,  # Set True if GPU supports it
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    print(f"\nTraining {mlm_config['model_name']}...")
    print(f"  Train: {len(train_ds)} sequences, Eval: {len(eval_ds)} sequences")
    print(f"  Epochs: {mlm_config['epochs']}, Batch: {mlm_config['batch_size']}")

    trainer.train(resume_from_checkpoint=args.resume)

    # Save final model
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))
    print(f"\nModel saved to {output_dir / 'final'}")


if __name__ == "__main__":
    main()
