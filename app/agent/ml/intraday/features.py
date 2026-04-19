"""Intraday feature engineering for 5-min bar pattern classification.

Produces 40+ features per bar:
- Time-of-day encoding (sin/cos)
- VWAP distance and slope
- Opening range position
- Volume metrics (imbalance, surge, cumulative)
- Lagged returns (1, 2, 3, 5, 10 bars)
- Intraday RSI, MACD, ATR, Bollinger %B
- Session stats (distance from day high/low, range used)
- Gap size
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Session constants
BARS_PER_SESSION = 78
OPENING_RANGE_BARS = 6


def build_intraday_features(
    df: pd.DataFrame,
    prev_close: Optional[float] = None,
) -> pd.DataFrame:
    """
    Build ML feature matrix from a single session's 5-min bars.

    Args:
        df: OHLCV DataFrame for one session. Columns: open, high, low, close, volume.
            Optional: vwap, date.
        prev_close: Previous session close for gap features.

    Returns:
        DataFrame with one row per bar, columns are features.
    """
    n = len(df)
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    opn = df["open"].values.astype(float)
    volume = df["volume"].values.astype(float)

    features = pd.DataFrame(index=df.index)

    # === 1. Time-of-day encoding (bar index 0–77) ===
    bar_idx = np.arange(n, dtype=float)
    features["bar_index"] = bar_idx
    features["bar_index_sin"] = np.sin(2 * np.pi * bar_idx / BARS_PER_SESSION)
    features["bar_index_cos"] = np.cos(2 * np.pi * bar_idx / BARS_PER_SESSION)
    features["is_opening_range"] = (bar_idx < OPENING_RANGE_BARS).astype(float)
    features["is_power_hour"] = (bar_idx >= 66).astype(float)
    features["is_lunch"] = ((bar_idx >= 24) & (bar_idx <= 42)).astype(float)

    # === 2. VWAP features ===
    if "vwap" in df.columns:
        vwap = df["vwap"].values.astype(float)
    else:
        tp = (high + low + close) / 3
        cum_tp_vol = np.cumsum(tp * volume)
        cum_vol = np.cumsum(volume)
        cum_vol[cum_vol == 0] = np.nan
        vwap = cum_tp_vol / cum_vol

    features["vwap_distance_pct"] = np.where(
        vwap != 0, (close - vwap) / vwap * 100, 0.0
    )
    features["above_vwap"] = (close > vwap).astype(float)
    # VWAP slope (5-bar diff)
    vwap_series = pd.Series(vwap)
    features["vwap_slope"] = vwap_series.diff(5).fillna(0).values

    # === 3. Opening range features ===
    or_high = high[:OPENING_RANGE_BARS].max() if n >= OPENING_RANGE_BARS else high[:n].max()
    or_low = low[:OPENING_RANGE_BARS].min() if n >= OPENING_RANGE_BARS else low[:n].min()
    or_range = or_high - or_low if or_high != or_low else 1e-6

    features["or_high_dist"] = (close - or_high) / or_range
    features["or_low_dist"] = (close - or_low) / or_range
    features["above_or_high"] = (close > or_high).astype(float)
    features["below_or_low"] = (close < or_low).astype(float)
    features["or_range_pct"] = or_range / close[0] if close[0] > 0 else 0.0

    # === 4. Volume features ===
    vol_cumsum = np.cumsum(volume)
    vol_mean = np.where(bar_idx > 0, vol_cumsum / (bar_idx + 1), volume)
    features["volume_ratio"] = np.where(vol_mean > 0, volume / vol_mean, 1.0)
    features["volume_surge"] = (features["volume_ratio"] > 1.5).astype(float)

    # Volume imbalance (up vs down volume over 10 bars)
    price_diff = np.diff(close, prepend=close[0])
    up_vol = np.where(price_diff > 0, volume, 0.0)
    down_vol = np.where(price_diff <= 0, volume, 0.0)
    up_vol_10 = pd.Series(up_vol).rolling(10, min_periods=1).sum().values
    down_vol_10 = pd.Series(down_vol).rolling(10, min_periods=1).sum().values
    total_vol_10 = up_vol_10 + down_vol_10
    features["volume_imbalance"] = np.where(
        total_vol_10 > 0, (up_vol_10 - down_vol_10) / total_vol_10, 0.0
    )

    # Cumulative volume percentile within session
    features["cum_volume_pct"] = np.where(
        vol_cumsum[-1] > 0, vol_cumsum / vol_cumsum[-1], 0.0
    ) if vol_cumsum[-1] > 0 else np.zeros(n)

    # === 5. Lagged returns ===
    close_series = pd.Series(close)
    for lag in [1, 2, 3, 5, 10]:
        features[f"return_{lag}bar"] = close_series.pct_change(lag).fillna(0).values

    # === 6. Intraday RSI(14) ===
    features["rsi_14"] = _compute_rsi(close, period=14)

    # === 7. Intraday MACD(12, 26, 9) ===
    ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().values
    ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().values
    macd_line = ema12 - ema26
    macd_signal = pd.Series(macd_line).ewm(span=9, adjust=False).mean().values
    features["macd"] = macd_line
    features["macd_signal"] = macd_signal
    features["macd_histogram"] = macd_line - macd_signal

    # === 8. Intraday ATR(14) ===
    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - np.roll(close, 1)),
            np.abs(low - np.roll(close, 1))
        )
    )
    tr[0] = high[0] - low[0]
    features["atr_14"] = pd.Series(tr).rolling(14, min_periods=1).mean().values
    features["atr_pct"] = np.where(close > 0, features["atr_14"] / close * 100, 0.0)

    # === 9. Bollinger %B (20-bar) ===
    close_roll = pd.Series(close)
    bb_mid = close_roll.rolling(20, min_periods=1).mean()
    bb_std = close_roll.rolling(20, min_periods=1).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_range = bb_upper - bb_lower
    features["bb_pct_b"] = np.where(
        bb_range > 0, (close - bb_lower.values) / bb_range.values, 0.5
    )
    features["bb_width"] = np.where(
        bb_mid > 0, bb_range.values / bb_mid.values, 0.0
    )

    # === 10. Session stats ===
    # Running high/low
    running_high = pd.Series(high).cummax().values
    running_low = pd.Series(low).cummin().values
    session_range = running_high - running_low
    features["dist_from_session_high"] = np.where(
        session_range > 0, (running_high - close) / session_range, 0.0
    )
    features["dist_from_session_low"] = np.where(
        session_range > 0, (close - running_low) / session_range, 0.0
    )
    features["range_used_pct"] = np.where(
        session_range > 0, (close - running_low) / session_range, 0.5
    )

    # === 11. Gap features ===
    if prev_close is not None and prev_close > 0:
        features["gap_pct"] = (opn[0] - prev_close) / prev_close
        features["gap_filled"] = np.where(
            opn[0] > prev_close,
            (close <= prev_close).astype(float),
            (close >= prev_close).astype(float),
        )
    else:
        features["gap_pct"] = 0.0
        features["gap_filled"] = 0.0

    # === 12. Bar-level features ===
    bar_range = high - low
    features["bar_range_pct"] = np.where(close > 0, bar_range / close * 100, 0.0)
    features["upper_shadow_pct"] = np.where(
        bar_range > 0, (high - np.maximum(opn, close)) / bar_range, 0.0
    )
    features["lower_shadow_pct"] = np.where(
        bar_range > 0, (np.minimum(opn, close) - low) / bar_range, 0.0
    )
    features["body_pct"] = np.where(
        bar_range > 0, np.abs(close - opn) / bar_range, 0.0
    )

    # Replace inf/nan
    features = features.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    logger.debug(f"Built {features.shape[1]} intraday features for {n} bars")
    return features


def build_multi_session_features(
    df: pd.DataFrame,
    session_breaks: List[int],
) -> pd.DataFrame:
    """
    Build features across multiple sessions.

    Args:
        df: Multi-session 5-min OHLCV DataFrame.
        session_breaks: List of indices where sessions start.

    Returns:
        Feature matrix covering all bars.
    """
    all_features = []
    prev_close = None

    for s_idx in range(len(session_breaks)):
        start = session_breaks[s_idx]
        end = session_breaks[s_idx + 1] if s_idx + 1 < len(session_breaks) else len(df)
        session_df = df.iloc[start:end]

        if len(session_df) < OPENING_RANGE_BARS:
            prev_close = session_df["close"].iloc[-1] if len(session_df) > 0 else prev_close
            continue

        features = build_intraday_features(session_df, prev_close=prev_close)
        all_features.append(features)
        prev_close = session_df["close"].iloc[-1]

    if not all_features:
        return pd.DataFrame()

    result = pd.concat(all_features)
    logger.info(f"Multi-session features: {result.shape[0]} bars x {result.shape[1]} features")
    return result


def _compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute RSI using Wilder's smoothing."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    rsi = np.full(len(close), 50.0)  # Default neutral
    if len(close) < period + 1:
        return rsi

    avg_gain = np.mean(gain[1:period + 1])
    avg_loss = np.mean(loss[1:period + 1])

    for i in range(period, len(close)):
        avg_gain = (avg_gain * (period - 1) + gain[i]) / period
        avg_loss = (avg_loss * (period - 1) + loss[i]) / period
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi
