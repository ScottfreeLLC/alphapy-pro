"""
Fetch OHLCV data from Massive API and save to CSV files.

Usage:
    python -m src.fetch_data --config config.yaml
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml

# Add backend to path for MassiveDataFetcher
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "app", "backend"))


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def fetch_daily_bars(fetcher, symbol: str, years: int = 5) -> pd.DataFrame:
    """Fetch daily bars for a symbol."""
    end = datetime.now()
    start = end - timedelta(days=years * 365)

    bars = fetcher.get_bars(
        symbol=symbol,
        timeframe="1d",
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
    )
    if not bars:
        return pd.DataFrame()

    df = pd.DataFrame(bars)
    return df


def fetch_intraday_bars(fetcher, symbol: str, months: int = 12) -> pd.DataFrame:
    """Fetch 5-min intraday bars for a symbol."""
    end = datetime.now()
    start = end - timedelta(days=months * 30)

    bars = fetcher.get_bars(
        symbol=symbol,
        timeframe="5min",
        start=start.strftime("%Y-%m-%d"),
        end=end.strftime("%Y-%m-%d"),
    )
    if not bars:
        return pd.DataFrame()

    df = pd.DataFrame(bars)
    return df


def main():
    parser = argparse.ArgumentParser(description="Fetch OHLCV data for Trade-GPT training")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--type", choices=["daily", "intraday", "both"], default="both")
    args = parser.parse_args()

    config = load_config(args.config)
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    from data_fetcher import MassiveDataFetcher
    fetcher = MassiveDataFetcher()

    if args.type in ("daily", "both"):
        daily_symbols = config["data"]["daily_symbols"]
        years = config["data"]["daily_years"]
        print(f"Fetching daily bars for {len(daily_symbols)} symbols ({years} years)...")

        for i, symbol in enumerate(daily_symbols):
            out_path = raw_dir / f"{symbol}_1d.csv"
            if out_path.exists():
                print(f"  [{i+1}/{len(daily_symbols)}] {symbol} — already exists, skipping")
                continue
            print(f"  [{i+1}/{len(daily_symbols)}] {symbol}...", end=" ", flush=True)
            df = fetch_daily_bars(fetcher, symbol, years)
            if len(df) > 0:
                df.to_csv(out_path, index=False)
                print(f"{len(df)} bars")
            else:
                print("no data")

    if args.type in ("intraday", "both"):
        intraday_symbols = config["data"]["intraday_symbols"]
        months = config["data"]["intraday_months"]
        print(f"\nFetching 5-min bars for {len(intraday_symbols)} symbols ({months} months)...")

        for i, symbol in enumerate(intraday_symbols):
            out_path = raw_dir / f"{symbol}_5min.csv"
            if out_path.exists():
                print(f"  [{i+1}/{len(intraday_symbols)}] {symbol} — already exists, skipping")
                continue
            print(f"  [{i+1}/{len(intraday_symbols)}] {symbol}...", end=" ", flush=True)
            df = fetch_intraday_bars(fetcher, symbol, months)
            if len(df) > 0:
                df.to_csv(out_path, index=False)
                print(f"{len(df)} bars")
            else:
                print("no data")


if __name__ == "__main__":
    main()
