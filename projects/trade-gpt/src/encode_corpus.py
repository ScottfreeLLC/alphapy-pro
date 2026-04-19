"""
Encode raw OHLCV CSVs into text token sequences for transformer training.

Usage:
    python -m src.encode_corpus --config config.yaml
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

# Add project root for PriceEncoder import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def encode_daily_file(csv_path: Path, encoder, out_dir: Path) -> str:
    """Encode a daily CSV file into grouped text (year > month markers)."""
    df = pd.read_csv(csv_path)

    # Ensure required columns
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            return ""
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    elif "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    elif "t" in df.columns:
        df["t"] = pd.to_datetime(df["t"])
        df = df.set_index("t")

    df = df.dropna(subset=["close"])
    if len(df) < 20:
        return ""

    encoded = encoder.encode_grouped(df, timeframe="1d")

    symbol = csv_path.stem.replace("_1d", "")
    out_path = out_dir / f"{symbol}_1d.txt"
    with open(out_path, "w") as f:
        f.write(encoded)

    return encoded


def encode_intraday_file(csv_path: Path, encoder, out_dir: Path) -> str:
    """Encode an intraday CSV file into grouped text (bod/eod markers)."""
    df = pd.read_csv(csv_path)

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in df.columns:
            return ""
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse timestamp
    for ts_col in ["timestamp", "t", "date"]:
        if ts_col in df.columns:
            df[ts_col] = pd.to_datetime(df[ts_col])
            df = df.set_index(ts_col)
            break

    df = df.dropna(subset=["close"])
    if len(df) < 20:
        return ""

    encoded = encoder.encode_grouped(df, timeframe="5min")

    symbol = csv_path.stem.replace("_5min", "")
    out_path = out_dir / f"{symbol}_5min.txt"
    with open(out_path, "w") as f:
        f.write(encoded)

    return encoded


def main():
    parser = argparse.ArgumentParser(description="Encode OHLCV CSVs into token corpus")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()

    config = load_config(args.config)
    period = config.get("encoding", {}).get("period", 20)

    from app.agent.ml.encoding import PriceEncoder
    encoder = PriceEncoder(period=period)

    raw_dir = Path("data/raw")
    encoded_dir = Path("data/encoded")
    encoded_dir.mkdir(parents=True, exist_ok=True)

    corpus_parts = []

    # Encode daily files
    daily_files = sorted(raw_dir.glob("*_1d.csv"))
    print(f"Encoding {len(daily_files)} daily files...")
    for i, f in enumerate(daily_files):
        print(f"  [{i+1}/{len(daily_files)}] {f.name}...", end=" ", flush=True)
        encoded = encode_daily_file(f, encoder, encoded_dir)
        if encoded:
            corpus_parts.append(encoded)
            tokens = len(encoded.split())
            print(f"{tokens} tokens")
        else:
            print("skipped")

    # Encode intraday files
    intraday_files = sorted(raw_dir.glob("*_5min.csv"))
    print(f"\nEncoding {len(intraday_files)} intraday files...")
    for i, f in enumerate(intraday_files):
        print(f"  [{i+1}/{len(intraday_files)}] {f.name}...", end=" ", flush=True)
        encoded = encode_intraday_file(f, encoder, encoded_dir)
        if encoded:
            corpus_parts.append(encoded)
            tokens = len(encoded.split())
            print(f"{tokens} tokens")
        else:
            print("skipped")

    # Write merged corpus
    corpus_path = encoded_dir / "corpus.txt"
    with open(corpus_path, "w") as f:
        f.write("\n".join(corpus_parts))

    total_tokens = sum(len(p.split()) for p in corpus_parts)
    print(f"\nCorpus written: {corpus_path} ({total_tokens:,} tokens, {len(corpus_parts)} documents)")


if __name__ == "__main__":
    main()
