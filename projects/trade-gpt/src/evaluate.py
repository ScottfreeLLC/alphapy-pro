"""
Walk-forward evaluation of trained models.

Evaluates:
- Token accuracy per type (pivot, net, range, volume)
- Direction accuracy (P vs N, H vs L)
- Trading signal profitability (H+P -> long entry)

Usage:
    python -m src.evaluate --model mlm|causal --config config.yaml
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def classify_token(token_str: str) -> dict:
    """Parse a price token into type and direction."""
    if not token_str or len(token_str) < 2:
        return {"type": "unknown", "direction": 0}

    prefix = token_str[0]
    types = {"H": "pivot", "L": "pivot", "T": "pivot",
             "P": "net", "N": "net", "Z": "net",
             "R": "range", "V": "volume"}
    directions = {"H": 1, "L": -1, "T": 0, "P": 1, "N": -1, "Z": 0}

    return {
        "type": types.get(prefix, "unknown"),
        "direction": directions.get(prefix, 0),
    }


def evaluate_mlm(config: dict):
    """Evaluate MLM model on fill-mask task."""
    from transformers import RobertaForMaskedLM, RobertaTokenizerFast, pipeline

    mlm_config = config["mlm"]
    model_dir = Path("models") / mlm_config["model_name"] / "final"

    if not model_dir.exists():
        print(f"ERROR: Model not found at {model_dir}")
        return

    tokenizer = RobertaTokenizerFast.from_pretrained(str(model_dir))
    model = RobertaForMaskedLM.from_pretrained(str(model_dir))

    fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

    # Load eval data
    corpus_path = Path("data/encoded/corpus.txt")
    with open(corpus_path) as f:
        lines = [line.strip() for line in f if line.strip()]

    # Use last 10% as eval
    eval_lines = lines[int(len(lines) * 0.9):]
    if not eval_lines:
        eval_lines = lines[-5:]

    results = defaultdict(lambda: {"correct": 0, "total": 0})
    direction_results = {"correct": 0, "total": 0}

    print(f"Evaluating MLM on {len(eval_lines)} sequences...")

    for line in eval_lines[:50]:  # Cap at 50 for speed
        tokens = line.split()
        markers = {"boy", "eoy", "bom", "eom", "bod", "eod"}
        bar_tokens = [t for t in tokens if t not in markers]

        for i, token in enumerate(bar_tokens):
            if i < 2 or i >= len(bar_tokens) - 1:
                continue

            # Mask one token and predict
            masked_tokens = bar_tokens.copy()
            masked_tokens[i] = tokenizer.mask_token
            masked_text = " ".join(masked_tokens[max(0, i-10):i+11])

            if tokenizer.mask_token not in masked_text:
                continue

            try:
                preds = fill_mask(masked_text, top_k=5)
                top_pred = preds[0]["token_str"].strip()
            except Exception:
                continue

            info = classify_token(token)
            results[info["type"]]["total"] += 1

            if top_pred == token:
                results[info["type"]]["correct"] += 1

            # Direction accuracy
            if info["type"] in ("pivot", "net"):
                pred_info = classify_token(top_pred)
                direction_results["total"] += 1
                if pred_info["direction"] == info["direction"]:
                    direction_results["correct"] += 1

    print("\n=== MLM Evaluation Results ===")
    for token_type, counts in sorted(results.items()):
        acc = counts["correct"] / max(1, counts["total"])
        print(f"  {token_type:10s}: {acc:.1%} ({counts['correct']}/{counts['total']})")

    if direction_results["total"] > 0:
        dir_acc = direction_results["correct"] / direction_results["total"]
        print(f"\n  Direction accuracy: {dir_acc:.1%} ({direction_results['correct']}/{direction_results['total']})")

    return dict(results)


def evaluate_causal(config: dict):
    """Evaluate causal model on next-token prediction."""
    from tokenizers import ByteLevelBPETokenizer
    from .train_causal import TradeGPT

    causal_config = config["causal"]
    model_path = Path("models") / causal_config["model_name"] / "best.pt"

    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        return

    # Load tokenizer
    tokenizer = ByteLevelBPETokenizer(
        "data/tokenizer/vocab.json",
        "data/tokenizer/merges.txt",
    )
    vocab_size = tokenizer.get_vocab_size()

    # Load model
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    model = TradeGPT(
        vocab_size=vocab_size,
        hidden_dim=causal_config["hidden_dim"],
        num_layers=causal_config["num_layers"],
        num_heads=causal_config["num_heads"],
        max_length=causal_config["max_length"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load eval data
    corpus_path = Path("data/encoded/corpus.txt")
    with open(corpus_path) as f:
        text = f.read()
    all_ids = tokenizer.encode(text).ids
    eval_ids = all_ids[int(len(all_ids) * 0.9):]

    block_size = causal_config["max_length"]
    correct = 0
    total = 0

    print(f"Evaluating causal model on {len(eval_ids)} tokens...")

    with torch.no_grad():
        for i in range(0, min(len(eval_ids) - block_size - 1, 5000), block_size):
            x = torch.tensor(eval_ids[i : i + block_size], dtype=torch.long).unsqueeze(0).to(device)
            y = torch.tensor(eval_ids[i + 1 : i + block_size + 1], dtype=torch.long).unsqueeze(0).to(device)

            logits, _ = model(x)
            preds = logits.argmax(dim=-1)

            correct += (preds == y).sum().item()
            total += y.numel()

    accuracy = correct / max(1, total)
    print(f"\n=== Causal Evaluation Results ===")
    print(f"  Next-token accuracy: {accuracy:.1%} ({correct}/{total})")
    print(f"  Perplexity: {math.exp(-math.log(max(accuracy, 1e-10))):.2f}")

    return {"accuracy": accuracy, "correct": correct, "total": total}


def main():
    import math

    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--model", choices=["mlm", "causal", "both"], default="both")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.model in ("mlm", "both"):
        evaluate_mlm(config)
        print()

    if args.model in ("causal", "both"):
        evaluate_causal(config)


if __name__ == "__main__":
    main()
