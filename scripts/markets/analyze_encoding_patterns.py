"""
Fetch recent price data, encode it, and find profitable repeatable patterns.

Analyzes which encoded token patterns consistently precede positive returns.
"""

import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# Backend imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app", "backend"))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "app", "backend", ".env"))

from data_fetcher import MassiveDataFetcher

# Encoding imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.agent.ml.encoding.encoder import encode_price_df
from app.agent.ml.encoding.patterns import find_patterns, get_pattern_catalog


SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "JPM", "V", "UNH", "HD", "NFLX", "ADBE", "CRM", "AMD",
    "PYPL", "BA", "GS", "XOM", "CVX", "COST", "COIN", "SQ",
]

LOOKBACK_DAYS = 365 * 2  # 2 years of daily data
FORWARD_RETURNS = [1, 3, 5, 10]  # Days ahead to measure returns


def fetch_symbol(fetcher, symbol, days_back):
    """Fetch daily bars for a symbol."""
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
    df = fetcher.fetch_bars(symbol, "1d", start, end)
    return df


def analyze_ngram_profitability(enc_df, original_df, n=2):
    """Find which n-gram token sequences precede profitable moves."""
    close = original_df["close"].values
    tokens = enc_df["encoded_str"].values

    ngram_stats = defaultdict(lambda: {
        "occurrences": 0,
        "returns_1d": [],
        "returns_3d": [],
        "returns_5d": [],
        "returns_10d": [],
    })

    for i in range(n - 1, len(tokens) - 10):
        ngram = " ".join(tokens[i - n + 1 : i + 1])
        stats = ngram_stats[ngram]
        stats["occurrences"] += 1

        for fwd in FORWARD_RETURNS:
            if i + fwd < len(close):
                ret = (close[i + fwd] - close[i]) / close[i]
                stats[f"returns_{fwd}d"].append(ret)

    return ngram_stats


def analyze_regex_patterns(enc_df, original_df):
    """Find which regex-defined patterns precede profitable moves."""
    close = original_df["close"].values
    tokens = enc_df["encoded_str"].values
    full_encoded = " ".join(tokens)

    pattern_stats = defaultdict(lambda: {
        "occurrences": 0,
        "returns_1d": [],
        "returns_3d": [],
        "returns_5d": [],
        "returns_10d": [],
    })

    # Sliding window analysis
    window = 5
    for i in range(window - 1, len(tokens) - 10):
        window_str = " ".join(tokens[i - window + 1 : i + 1])
        matches = find_patterns(window_str)

        for m in matches:
            stats = pattern_stats[m.name]
            stats["occurrences"] += 1

            for fwd in FORWARD_RETURNS:
                if i + fwd < len(close):
                    ret = (close[i + fwd] - close[i]) / close[i]
                    stats[f"returns_{fwd}d"].append(ret)

    return pattern_stats


def print_top_patterns(ngram_stats, title, min_occurrences=10, sort_by="returns_5d"):
    """Print the most profitable repeating patterns."""
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}")

    # Filter and compute stats
    rows = []
    for pattern, stats in ngram_stats.items():
        if stats["occurrences"] < min_occurrences:
            continue
        row = {"pattern": pattern, "count": stats["occurrences"]}
        for fwd in FORWARD_RETURNS:
            rets = stats[f"returns_{fwd}d"]
            if rets:
                row[f"avg_{fwd}d"] = np.mean(rets) * 100
                row[f"win_{fwd}d"] = (np.array(rets) > 0).mean() * 100
                row[f"sharpe_{fwd}d"] = (np.mean(rets) / max(np.std(rets), 1e-10)) * np.sqrt(252 / fwd)
            else:
                row[f"avg_{fwd}d"] = 0
                row[f"win_{fwd}d"] = 50
                row[f"sharpe_{fwd}d"] = 0
        rows.append(row)

    if not rows:
        print("  No patterns with enough occurrences found.")
        return []

    # Sort by average return
    sort_key = f"avg_{sort_by.replace('returns_', '').replace('d', '')}d" if "returns" in sort_by else f"avg_{sort_by}"
    rows.sort(key=lambda r: r.get(sort_key, 0), reverse=True)

    # Print top bullish
    print(f"\n  TOP BULLISH PATTERNS (sorted by avg 5d return)")
    print(f"  {'Pattern':<35} {'Count':>6} {'Avg 1d':>8} {'Avg 5d':>8} {'Win% 5d':>8} {'Sharpe':>8}")
    print(f"  {'-'*35} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for row in rows[:15]:
        print(f"  {row['pattern']:<35} {row['count']:>6} {row['avg_1d']:>7.2f}% {row['avg_5d']:>7.2f}% {row['win_5d']:>7.1f}% {row.get('sharpe_5d', 0):>8.2f}")

    # Print top bearish
    print(f"\n  TOP BEARISH PATTERNS (sorted by avg 5d return)")
    print(f"  {'Pattern':<35} {'Count':>6} {'Avg 1d':>8} {'Avg 5d':>8} {'Win% 5d':>8} {'Sharpe':>8}")
    print(f"  {'-'*35} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for row in rows[-15:]:
        print(f"  {row['pattern']:<35} {row['count']:>6} {row['avg_1d']:>7.2f}% {row['avg_5d']:>7.2f}% {row['win_5d']:>7.1f}% {row.get('sharpe_5d', 0):>8.2f}")

    return rows


def main():
    fetcher = MassiveDataFetcher()

    all_ngram_stats = defaultdict(lambda: {
        "occurrences": 0,
        "returns_1d": [],
        "returns_3d": [],
        "returns_5d": [],
        "returns_10d": [],
    })
    all_pattern_stats = defaultdict(lambda: {
        "occurrences": 0,
        "returns_1d": [],
        "returns_3d": [],
        "returns_5d": [],
        "returns_10d": [],
    })

    print(f"Fetching {LOOKBACK_DAYS} days of daily data for {len(SYMBOLS)} symbols...")
    print()

    for i, symbol in enumerate(SYMBOLS):
        print(f"  [{i+1}/{len(SYMBOLS)}] {symbol}...", end=" ", flush=True)
        df = fetch_symbol(fetcher, symbol, LOOKBACK_DAYS)

        if df is None or df.empty or len(df) < 60:
            print("insufficient data")
            continue

        # Ensure numeric
        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
        df = df.dropna(subset=["close"]).reset_index(drop=True)

        print(f"{len(df)} bars", end=" ", flush=True)

        # Encode
        enc_df = encode_price_df(df, p=20)
        print(f"-> {len(enc_df)} encoded", end=" ", flush=True)

        # N-gram analysis (2-grams and 3-grams)
        for n in [2, 3]:
            ngram_stats = analyze_ngram_profitability(enc_df, df, n=n)
            for pattern, stats in ngram_stats.items():
                merged = all_ngram_stats[pattern]
                merged["occurrences"] += stats["occurrences"]
                for fwd in FORWARD_RETURNS:
                    merged[f"returns_{fwd}d"].extend(stats[f"returns_{fwd}d"])

        # Regex pattern analysis
        pattern_stats = analyze_regex_patterns(enc_df, df)
        for pattern, stats in pattern_stats.items():
            merged = all_pattern_stats[pattern]
            merged["occurrences"] += stats["occurrences"]
            for fwd in FORWARD_RETURNS:
                merged[f"returns_{fwd}d"].extend(stats[f"returns_{fwd}d"])

        print("done")

    # Print results
    print_top_patterns(
        all_ngram_stats,
        "N-GRAM TOKEN PATTERNS (across all symbols)",
        min_occurrences=50,
        sort_by="5d",
    )

    print_top_patterns(
        all_pattern_stats,
        "REGEX-DEFINED PATTERNS (across all symbols)",
        min_occurrences=20,
        sort_by="5d",
    )

    # Summary stats
    print(f"\n{'='*80}")
    print(f" SUMMARY")
    print(f"{'='*80}")
    total_ngrams = len([k for k, v in all_ngram_stats.items() if v["occurrences"] >= 50])
    total_patterns = len([k for k, v in all_pattern_stats.items() if v["occurrences"] >= 20])
    print(f"  N-grams with 50+ occurrences: {total_ngrams}")
    print(f"  Regex patterns with 20+ occurrences: {total_patterns}")

    # Best overall signals
    print(f"\n  BEST ACTIONABLE SIGNALS (>=50 occurrences, >55% win rate, >0.3% avg 5d return):")
    best = []
    for pattern, stats in {**all_ngram_stats, **all_pattern_stats}.items():
        if stats["occurrences"] < 50:
            continue
        rets_5d = stats["returns_5d"]
        if not rets_5d:
            continue
        avg = np.mean(rets_5d) * 100
        win = (np.array(rets_5d) > 0).mean() * 100
        if win > 55 and avg > 0.3:
            best.append((pattern, stats["occurrences"], avg, win))

    best.sort(key=lambda x: x[2], reverse=True)
    for pat, count, avg, win in best[:20]:
        print(f"    {pat:<35} count={count:>5}  avg_5d={avg:>6.2f}%  win={win:>5.1f}%")

    if not best:
        print("    (none found — try with more data or lower thresholds)")


if __name__ == "__main__":
    main()
