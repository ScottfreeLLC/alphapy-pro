"""
Deep analysis of encoded price patterns — looks at component-level patterns,
conditional probabilities, and regime-aware signals.
"""

import os
import sys
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app", "backend"))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "app", "backend", ".env"))
from data_fetcher import MassiveDataFetcher

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.agent.ml.encoding.encoder import encode_price_df
from app.agent.ml.encoding.price_encoder import PriceEncoder


SYMBOLS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
    "JPM", "V", "UNH", "HD", "NFLX", "ADBE", "CRM", "AMD",
    "PYPL", "BA", "GS", "XOM", "CVX", "COST", "COIN", "SQ",
]

FORWARD_DAYS = [1, 3, 5, 10]


def fetch_all(fetcher, symbols, days_back=730):
    """Fetch daily bars for all symbols."""
    data = {}
    end = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

    for i, sym in enumerate(symbols):
        print(f"  [{i+1}/{len(symbols)}] {sym}...", end=" ", flush=True)
        df = fetcher.fetch_bars(sym, "1d", start, end)
        if df is not None and not df.empty and len(df) >= 60:
            for col in ["open", "high", "low", "close"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
            df = df.dropna(subset=["close"]).reset_index(drop=True)
            data[sym] = df
            print(f"{len(df)} bars")
        else:
            print("skip")
    return data


def build_encoded_dataset(all_data):
    """Encode all symbols and build a unified dataset with forward returns."""
    rows = []
    for sym, df in all_data.items():
        enc = encode_price_df(df, p=20)
        close = df["close"].values

        for i in range(20, len(df) - 10):
            row = {
                "symbol": sym,
                "bar_idx": i,
                "close": close[i],
                # Encoding components
                "token": enc["encoded_str"].iloc[i],
                "pivot_str": enc["pivot_str"].iloc[i],
                "net_str": enc["net_str"].iloc[i],
                "range_str": enc["range_str"].iloc[i],
                "volume_str": enc["volume_str"].iloc[i],
                # Numeric features
                "pivot_strength": enc["pivot_strength"].iloc[i],
                "pivot_direction": enc["pivot_direction"].iloc[i],
                "net_direction": enc["net_direction"].iloc[i],
                "net_magnitude": enc["net_magnitude"].iloc[i],
                "range_magnitude": enc["range_magnitude"].iloc[i],
                "volume_magnitude": enc["volume_magnitude"].iloc[i],
                # Previous bar context
                "prev_token": enc["encoded_str"].iloc[i-1] if i > 0 else "",
                "prev_pivot": enc["pivot_str"].iloc[i-1] if i > 0 else "",
                "prev_net": enc["net_str"].iloc[i-1] if i > 0 else "",
                # 2-gram
                "bigram": f"{enc['encoded_str'].iloc[i-1]} {enc['encoded_str'].iloc[i]}",
                # 3-gram
                "trigram": f"{enc['encoded_str'].iloc[i-2]} {enc['encoded_str'].iloc[i-1]} {enc['encoded_str'].iloc[i]}" if i >= 2 else "",
            }
            # Forward returns
            for fwd in FORWARD_DAYS:
                if i + fwd < len(close):
                    row[f"fwd_{fwd}d"] = (close[i + fwd] - close[i]) / close[i]
                else:
                    row[f"fwd_{fwd}d"] = np.nan
            rows.append(row)

    return pd.DataFrame(rows)


def analyze_single_dimension(dataset, dim_col, dim_name, min_count=100):
    """Analyze forward returns by a single encoding dimension."""
    print(f"\n{'─'*70}")
    print(f"  {dim_name.upper()} DIMENSION")
    print(f"{'─'*70}")

    groups = dataset.groupby(dim_col)
    results = []
    for val, grp in groups:
        if len(grp) < min_count:
            continue
        row = {"value": val, "count": len(grp)}
        for fwd in FORWARD_DAYS:
            col = f"fwd_{fwd}d"
            rets = grp[col].dropna()
            row[f"avg_{fwd}d"] = rets.mean() * 100
            row[f"win_{fwd}d"] = (rets > 0).mean() * 100
            row[f"sharpe_{fwd}d"] = (rets.mean() / max(rets.std(), 1e-10)) * np.sqrt(252/fwd)
        results.append(row)

    results.sort(key=lambda r: r.get("avg_5d", 0), reverse=True)

    print(f"  {'Value':<12} {'Count':>6}  {'Avg1d':>7} {'Avg5d':>7} {'Win5d':>6} {'Shp5d':>6}  {'Avg10d':>7} {'Win10d':>6}")
    print(f"  {'─'*12} {'─'*6}  {'─'*7} {'─'*7} {'─'*6} {'─'*6}  {'─'*7} {'─'*6}")
    for r in results:
        print(f"  {r['value']:<12} {r['count']:>6}  {r['avg_1d']:>6.2f}% {r['avg_5d']:>6.2f}% {r['win_5d']:>5.1f}% {r['sharpe_5d']:>6.2f}  {r['avg_10d']:>6.2f}% {r['win_10d']:>5.1f}%")


def analyze_combinations(dataset, min_count=30):
    """Analyze pivot + net combinations — most actionable signals."""
    print(f"\n{'='*70}")
    print(f"  PIVOT + NET COMBINATIONS (the money patterns)")
    print(f"{'='*70}")

    dataset["pivot_net"] = dataset["pivot_str"] + " " + dataset["net_str"]
    groups = dataset.groupby("pivot_net")

    results = []
    for val, grp in groups:
        if len(grp) < min_count:
            continue
        rets_5 = grp["fwd_5d"].dropna()
        rets_10 = grp["fwd_10d"].dropna()
        row = {
            "combo": val,
            "count": len(grp),
            "avg_5d": rets_5.mean() * 100,
            "win_5d": (rets_5 > 0).mean() * 100,
            "sharpe_5d": (rets_5.mean() / max(rets_5.std(), 1e-10)) * np.sqrt(252/5),
            "avg_10d": rets_10.mean() * 100,
            "win_10d": (rets_10 > 0).mean() * 100,
        }
        results.append(row)

    results.sort(key=lambda r: r["avg_5d"], reverse=True)

    print(f"\n  {'Pivot+Net':<12} {'Count':>6}  {'Avg5d':>7} {'Win5d':>6} {'Shp5d':>6}  {'Avg10d':>7} {'Win10d':>6}")
    print(f"  {'─'*12} {'─'*6}  {'─'*7} {'─'*6} {'─'*6}  {'─'*7} {'─'*6}")
    for r in results:
        # Highlight strong signals
        marker = ""
        if r["win_5d"] > 57 and r["avg_5d"] > 0.3:
            marker = " <-- BULLISH"
        elif r["win_5d"] < 43 and r["avg_5d"] < -0.3:
            marker = " <-- BEARISH"
        print(f"  {r['combo']:<12} {r['count']:>6}  {r['avg_5d']:>6.2f}% {r['win_5d']:>5.1f}% {r['sharpe_5d']:>6.2f}  {r['avg_10d']:>6.2f}% {r['win_10d']:>5.1f}%{marker}")


def analyze_sequences(dataset, min_count=20):
    """Analyze 2-bar sequences: what follows what?"""
    print(f"\n{'='*70}")
    print(f"  SEQUENTIAL PATTERNS: 'After X, if Y appears...'")
    print(f"{'='*70}")

    # Focus: after a pivot low + negative bar, what predicts reversal?
    bearish_bars = dataset[dataset["net_str"].isin(["N1", "N2"])]
    next_bar_positive = bearish_bars[bearish_bars["pivot_strength"] >= 5]

    if len(next_bar_positive) >= 10:
        rets = next_bar_positive["fwd_5d"].dropna()
        print(f"\n  After strong pivot low (L5+) with negative net:")
        print(f"    Count: {len(next_bar_positive)}")
        print(f"    Avg 5d return: {rets.mean()*100:.2f}%")
        print(f"    Win rate 5d: {(rets > 0).mean()*100:.1f}%")

    # After volume climax (V2) with negative net
    climax = dataset[(dataset["volume_str"] == "V2") & (dataset["net_direction"] == -1)]
    if len(climax) >= 10:
        rets = climax["fwd_5d"].dropna()
        print(f"\n  After volume climax (V2) on down bar:")
        print(f"    Count: {len(climax)}")
        print(f"    Avg 5d return: {rets.mean()*100:.2f}%")
        print(f"    Win rate 5d: {(rets > 0).mean()*100:.1f}%")

    # H20 (max pivot high) — trend strength
    h20 = dataset[dataset["pivot_str"] == "H20"]
    if len(h20) >= 10:
        rets = h20["fwd_5d"].dropna()
        print(f"\n  At H20 (20-bar high streak):")
        print(f"    Count: {len(h20)}")
        print(f"    Avg 5d return: {rets.mean()*100:.2f}%")
        print(f"    Win rate 5d: {(rets > 0).mean()*100:.1f}%")

    # Best bigrams
    print(f"\n  TOP 15 BIGRAMS BY 5-DAY RETURN (min 20 occurrences):")
    bigram_groups = dataset.groupby("bigram")
    bigram_results = []
    for bg, grp in bigram_groups:
        if len(grp) < 20:
            continue
        rets = grp["fwd_5d"].dropna()
        bigram_results.append({
            "bigram": bg,
            "count": len(grp),
            "avg_5d": rets.mean() * 100,
            "win_5d": (rets > 0).mean() * 100,
        })

    bigram_results.sort(key=lambda r: r["avg_5d"], reverse=True)
    print(f"  {'Bigram':<35} {'Count':>6} {'Avg5d':>7} {'Win5d':>6}")
    print(f"  {'─'*35} {'─'*6} {'─'*7} {'─'*6}")
    for r in bigram_results[:15]:
        print(f"  {r['bigram']:<35} {r['count']:>6} {r['avg_5d']:>6.2f}% {r['win_5d']:>5.1f}%")

    print(f"\n  BOTTOM 15 BIGRAMS (bearish):")
    for r in bigram_results[-15:]:
        print(f"  {r['bigram']:<35} {r['count']:>6} {r['avg_5d']:>6.2f}% {r['win_5d']:>5.1f}%")


def analyze_high_volume_context(dataset, min_count=20):
    """High volume bars are the most informative — analyze separately."""
    print(f"\n{'='*70}")
    print(f"  HIGH VOLUME (V2) BAR CONTEXT")
    print(f"{'='*70}")

    hv = dataset[dataset["volume_str"] == "V2"]
    print(f"  Total V2 bars: {len(hv)}")

    # By pivot direction + net on high volume
    for pivot_dir, label in [(1, "pivot HIGH"), (-1, "pivot LOW"), (0, "tied pivot")]:
        for net_dir, net_label in [(1, "positive net"), (-1, "negative net")]:
            subset = hv[(hv["pivot_direction"] == pivot_dir) & (hv["net_direction"] == net_dir)]
            if len(subset) < min_count:
                continue
            rets = subset["fwd_5d"].dropna()
            if len(rets) == 0:
                continue
            print(f"\n  V2 + {label} + {net_label}:")
            print(f"    Count: {len(subset)}  Avg 5d: {rets.mean()*100:.2f}%  Win: {(rets>0).mean()*100:.1f}%")


def main():
    fetcher = MassiveDataFetcher()

    print("Fetching 2 years of daily data...")
    all_data = fetch_all(fetcher, SYMBOLS)

    print(f"\nBuilding encoded dataset...")
    dataset = build_encoded_dataset(all_data)
    print(f"Dataset: {len(dataset)} rows, {dataset['symbol'].nunique()} symbols")

    # Dimension-by-dimension analysis
    analyze_single_dimension(dataset, "pivot_str", "pivot", min_count=50)
    analyze_single_dimension(dataset, "net_str", "net change", min_count=50)
    analyze_single_dimension(dataset, "range_str", "range", min_count=50)
    analyze_single_dimension(dataset, "volume_str", "volume", min_count=50)

    # Combination analysis
    analyze_combinations(dataset, min_count=30)

    # Sequential patterns
    analyze_sequences(dataset, min_count=20)

    # High volume context
    analyze_high_volume_context(dataset, min_count=20)


if __name__ == "__main__":
    main()
