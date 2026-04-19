"""
Standalone price encoder — reimplemented from alphapy-pro/alphapy/nlp.py.

Converts OHLCV bars into text tokens encoding pivot strength, net change,
range, and volume relative to their moving averages.

Vocabulary:
  Pivot:  H1-H20 (pivot high) / L1-L20 (pivot low) / T0 (tied)
  Net:    P0/P1/P2 (positive) / N0/N1/N2 (negative) / Z0 (zero)
  Range:  R0/R1/R2
  Volume: V0/V1/V2
"""

import math

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Pivot helpers
# ---------------------------------------------------------------------------

def pivot_high(df: pd.DataFrame, col: str = "high", p: int = 20) -> pd.Series:
    """Count consecutive bars where current value is the highest (up to *p*)."""

    def _count(ds: pd.Series, max_len: int) -> int:
        ds_len = min(len(ds), max_len)
        if ds_len == 1:
            return 1
        window = ds.iloc[-ds_len:].values
        value = window[-1]
        count = 1
        for i in range(len(window) - 2, -1, -1):
            if value > window[i]:
                count += 1
            else:
                break
        return count

    result = df[col].expanding().apply(lambda s: _count(s, p), raw=False)
    return result.astype(int)


def pivot_low(df: pd.DataFrame, col: str = "low", p: int = 20) -> pd.Series:
    """Count consecutive bars where current value is the lowest (up to *p*)."""

    def _count(ds: pd.Series, max_len: int) -> int:
        ds_len = min(len(ds), max_len)
        if ds_len == 1:
            return 1
        window = ds.iloc[-ds_len:].values
        value = window[-1]
        count = 1
        for i in range(len(window) - 2, -1, -1):
            if value < window[i]:
                count += 1
            else:
                break
        return count

    result = df[col].expanding().apply(lambda s: _count(s, p), raw=False)
    return result.astype(int)


# ---------------------------------------------------------------------------
# Token encoders (row-level)
# ---------------------------------------------------------------------------

def encode_pivot(row: pd.Series, c_high: str, c_low: str) -> str:
    """Encode strongest pivot direction: H{n} / L{n} / T0."""
    h = row[c_high]
    l = row[c_low]
    if h > l:
        return f"H{int(h)}"
    elif l > h:
        return f"L{int(l)}"
    return "T0"


def encode_net(row: pd.Series, net_col: str, ma_col: str) -> str:
    """Encode net change relative to MA(|net|): P/N/Z + magnitude 0-2."""
    net_val = row[net_col]
    ma_val = row[ma_col]
    if net_val > 0:
        try:
            mag = min(int(net_val // ma_val), 2)
            if math.isnan(mag):
                mag = 0
        except (ZeroDivisionError, ValueError):
            mag = 0
        return f"P{mag}"
    elif net_val < 0:
        try:
            mag = min(int(abs((net_val // ma_val) + 1)), 2)
            if math.isnan(mag):
                mag = 0
        except (ZeroDivisionError, ValueError):
            mag = 0
        return f"N{mag}"
    return "Z0"


def encode_range(row: pd.Series, range_col: str, ma_col: str) -> str:
    """Encode bar range relative to MA(range): R + 0-2."""
    try:
        mag = min(int(row[range_col] // row[ma_col]), 2)
        if math.isnan(mag):
            mag = 0
    except (ZeroDivisionError, ValueError):
        mag = 0
    return f"R{mag}"


def encode_volume(row: pd.Series, vol_col: str, ma_col: str) -> str:
    """Encode volume relative to MA(volume): V + 0-2."""
    try:
        mag = min(int(row[vol_col] // row[ma_col]), 2)
        if math.isnan(mag):
            mag = 0
    except (ZeroDivisionError, ValueError):
        mag = 0
    return f"V{mag}"


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def encode_price(
    df: pd.DataFrame,
    p: int = 20,
    intraday: tuple | None = None,
) -> str:
    """Encode an OHLCV DataFrame into a space-separated token string.

    Parameters
    ----------
    df : DataFrame
        Must contain columns: open, high, low, close, volume.
    p : int
        Lookback period for moving averages and pivot counting.
    intraday : tuple or None
        If provided, (start_marker, end_marker) to wrap the sequence,
        e.g. ("bod", "eod") for daily sessions.

    Returns
    -------
    str
        Space-separated encoded tokens, e.g. "H3P1R0V2 L1N0R1V0 ..."
    """
    df = df.copy()

    # Pivot strength
    df["_pivot_high"] = pivot_high(df, "high", p)
    df["_pivot_low"] = pivot_low(df, "low", p)
    df["_pivot_str"] = df.apply(encode_pivot, args=("_pivot_high", "_pivot_low"), axis=1)

    # Net change
    df["_net"] = df["close"] - df["close"].shift(1)
    df["_net_abs"] = df["_net"].abs()
    df["_net_ma"] = df["_net_abs"].rolling(p).mean()
    df["_net_str"] = df.apply(encode_net, args=("_net", "_net_ma"), axis=1)

    # Range
    df["_range"] = df["high"] - df["low"]
    df["_range_ma"] = df["_range"].rolling(p).mean()
    df["_range_str"] = df.apply(encode_range, args=("_range", "_range_ma"), axis=1)

    # Volume
    df["_vol_ma"] = df["volume"].rolling(p).mean()
    df["_vol_str"] = df.apply(encode_volume, args=("volume", "_vol_ma"), axis=1)

    # Combine tokens
    df["_encoded"] = df["_pivot_str"] + df["_net_str"] + df["_range_str"] + df["_vol_str"]
    encoded_str = " ".join(df["_encoded"].values)

    if intraday:
        encoded_str = f"{intraday[0]} {encoded_str} {intraday[1]}"

    return encoded_str


def encode_price_df(
    df: pd.DataFrame,
    p: int = 20,
) -> pd.DataFrame:
    """Encode OHLCV and return a DataFrame with per-bar encoding columns.

    Returns the original DataFrame with added columns:
      pivot_str, net_str, range_str, volume_str, encoded_str,
      pivot_strength, pivot_direction, net_direction, net_magnitude,
      range_magnitude, volume_magnitude
    """
    df = df.copy()

    # Pivot
    df["pivot_high"] = pivot_high(df, "high", p)
    df["pivot_low"] = pivot_low(df, "low", p)
    df["pivot_str"] = df.apply(encode_pivot, args=("pivot_high", "pivot_low"), axis=1)

    # Numeric pivot features
    df["pivot_strength"] = df[["pivot_high", "pivot_low"]].max(axis=1)
    df["pivot_direction"] = np.where(
        df["pivot_high"] > df["pivot_low"], 1,
        np.where(df["pivot_low"] > df["pivot_high"], -1, 0)
    )

    # Net
    df["net"] = df["close"] - df["close"].shift(1)
    df["net_abs"] = df["net"].abs()
    df["net_ma"] = df["net_abs"].rolling(p).mean()
    df["net_str"] = df.apply(encode_net, args=("net", "net_ma"), axis=1)

    # Numeric net features
    df["net_direction"] = np.sign(df["net"]).fillna(0).astype(int)
    net_ma_safe = df["net_ma"].replace(0, np.nan)
    df["net_magnitude"] = (df["net_abs"] / net_ma_safe).clip(upper=2).fillna(0)

    # Range
    df["range"] = df["high"] - df["low"]
    df["range_ma"] = df["range"].rolling(p).mean()
    df["range_str"] = df.apply(encode_range, args=("range", "range_ma"), axis=1)

    range_ma_safe = df["range_ma"].replace(0, np.nan)
    df["range_magnitude"] = (df["range"] / range_ma_safe).clip(upper=2).fillna(0)

    # Volume
    df["vol_ma"] = df["volume"].rolling(p).mean()
    df["volume_str"] = df.apply(encode_volume, args=("volume", "vol_ma"), axis=1)

    vol_ma_safe = df["vol_ma"].replace(0, np.nan)
    df["volume_magnitude"] = (df["volume"] / vol_ma_safe).clip(upper=2).fillna(0)

    # Combined token
    df["encoded_str"] = df["pivot_str"] + df["net_str"] + df["range_str"] + df["volume_str"]

    return df
