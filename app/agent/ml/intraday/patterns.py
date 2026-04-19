"""Intraday pattern definitions and rule-based heuristic labeler.

Defines 8 canonical intraday patterns + a NO_PATTERN background class.
The heuristic labeler assigns labels to historical 5-min bars for training
the XGBoost classifier. Labels are *noisy* by design — the classifier learns
to generalize beyond the heuristics.
"""

import logging
from enum import IntEnum
from typing import Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Bars per trading session (6.5 hours @ 5-min bars = 78 bars, indices 0–77)
BARS_PER_SESSION = 78

# Opening range = first 30 min = 6 bars (indices 0–5)
OPENING_RANGE_BARS = 6

# Power hour starts at bar index 66 (3:00 PM ET = last 60 min)
POWER_HOUR_START = 66


class IntradayPattern(IntEnum):
    """Canonical intraday patterns."""
    NO_PATTERN = 0
    ORB_BREAKOUT = 1        # Opening Range Breakout
    MORNING_REVERSAL = 2    # Reversal of gap / opening move in first 90 min
    VWAP_RECLAIM = 3        # Price reclaims VWAP after trading below
    GAP_FILL = 4            # Price fills overnight gap
    POWER_HOUR = 5          # Strong directional move in final hour
    MEAN_REVERSION = 6      # Extreme RSI / BB reversion
    MOMENTUM_BREAKOUT = 7   # Volume surge + new session high/low
    RANGE_EXPANSION = 8     # Bollinger squeeze → expansion


PATTERN_NAMES = {p: p.name for p in IntradayPattern}
NUM_CLASSES = len(IntradayPattern)


def label_session(df: pd.DataFrame, prev_close: float = None) -> pd.Series:
    """
    Label each bar in a single-session 5-min DataFrame with a pattern.

    Args:
        df: OHLCV DataFrame for one trading session (up to 78 bars).
            Must have columns: open, high, low, close, volume.
            Optionally: vwap.
        prev_close: Previous session's closing price (for gap detection).

    Returns:
        Series of IntradayPattern int labels, same index as df.
    """
    labels = pd.Series(IntradayPattern.NO_PATTERN, index=df.index, dtype=int)
    n = len(df)
    if n < OPENING_RANGE_BARS + 2:
        return labels

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values.astype(float)
    opn = df["open"].values

    # Session stats
    or_high = high[:OPENING_RANGE_BARS].max()
    or_low = low[:OPENING_RANGE_BARS].min()
    session_open = opn[0]
    vol_mean = volume.mean() if volume.mean() > 0 else 1.0

    # VWAP (use column if available, else compute)
    if "vwap" in df.columns:
        vwap = df["vwap"].values
    else:
        tp = (high + low + close) / 3
        cum_tp_vol = np.cumsum(tp * volume)
        cum_vol = np.cumsum(volume)
        cum_vol[cum_vol == 0] = np.nan
        vwap = cum_tp_vol / cum_vol

    # Gap size
    gap_pct = 0.0
    if prev_close is not None and prev_close > 0:
        gap_pct = (session_open - prev_close) / prev_close

    # --- Compute indicators used across patterns ---
    # 14-bar RSI
    rsi = _fast_rsi(close, period=14)

    # Bollinger %B (20-bar)
    bb_pct_b = _fast_bb_pct_b(close, period=20)

    # --- Label each bar ---
    for i in range(n):
        bar_idx = i  # 0-based index within session

        # 1. ORB Breakout (bars 6–20, price breaks OR range on volume)
        if OPENING_RANGE_BARS <= bar_idx <= 20:
            vol_surge = volume[i] > 1.5 * vol_mean
            if close[i] > or_high and vol_surge:
                labels.iloc[i] = IntradayPattern.ORB_BREAKOUT
                continue
            if close[i] < or_low and vol_surge:
                labels.iloc[i] = IntradayPattern.ORB_BREAKOUT
                continue

        # 2. Morning Reversal (bars 6–18, reverses opening direction)
        if OPENING_RANGE_BARS <= bar_idx <= 18:
            open_move = close[OPENING_RANGE_BARS - 1] - session_open
            bar_move = close[i] - close[max(0, i - 3)]
            # Reversal: opening was up, now pulling back (or vice versa)
            if abs(open_move) > 0 and np.sign(bar_move) != np.sign(open_move):
                if abs(bar_move) > 0.3 * abs(open_move):
                    labels.iloc[i] = IntradayPattern.MORNING_REVERSAL
                    continue

        # 3. VWAP Reclaim (price crosses from below to above VWAP)
        if i >= 2 and not np.isnan(vwap[i]):
            was_below = close[i - 1] < vwap[i - 1] and close[i - 2] < vwap[i - 2]
            now_above = close[i] > vwap[i]
            if was_below and now_above and volume[i] > 1.2 * vol_mean:
                labels.iloc[i] = IntradayPattern.VWAP_RECLAIM
                continue

        # 4. Gap Fill (price fills overnight gap within session)
        if prev_close is not None and abs(gap_pct) > 0.005:
            if gap_pct > 0 and close[i] <= prev_close and i <= 40:
                labels.iloc[i] = IntradayPattern.GAP_FILL
                continue
            if gap_pct < 0 and close[i] >= prev_close and i <= 40:
                labels.iloc[i] = IntradayPattern.GAP_FILL
                continue

        # 5. Power Hour (strong directional move in final 60 min)
        if bar_idx >= POWER_HOUR_START and i >= 3:
            ph_return = (close[i] - close[POWER_HOUR_START]) / close[POWER_HOUR_START]
            vol_above_avg = volume[i] > vol_mean
            if abs(ph_return) > 0.005 and vol_above_avg:
                labels.iloc[i] = IntradayPattern.POWER_HOUR
                continue

        # 6. Mean Reversion (extreme RSI + Bollinger)
        if i >= 20:
            rsi_val = rsi[i]
            bb_val = bb_pct_b[i]
            if not np.isnan(rsi_val) and not np.isnan(bb_val):
                if (rsi_val < 25 and bb_val < 0.05) or (rsi_val > 75 and bb_val > 0.95):
                    labels.iloc[i] = IntradayPattern.MEAN_REVERSION
                    continue

        # 7. Momentum Breakout (new session high/low with volume surge)
        if i >= OPENING_RANGE_BARS:
            is_new_high = close[i] >= high[:i].max()
            is_new_low = close[i] <= low[:i].min()
            vol_surge = volume[i] > 2.0 * vol_mean
            if (is_new_high or is_new_low) and vol_surge:
                labels.iloc[i] = IntradayPattern.MOMENTUM_BREAKOUT
                continue

        # 8. Range Expansion (Bollinger squeeze → expand)
        if i >= 25:
            bb_width_recent = bb_pct_b[i] - bb_pct_b[i - 5] if not np.isnan(bb_pct_b[i - 5]) else 0
            # Simple proxy: narrow bands followed by large bar
            bar_range_pct = (high[i] - low[i]) / close[i] if close[i] > 0 else 0
            avg_bar_range = np.mean([(high[j] - low[j]) / close[j] for j in range(max(0, i - 10), i) if close[j] > 0]) if i > 0 else 0
            if bar_range_pct > 2.0 * avg_bar_range and avg_bar_range > 0:
                labels.iloc[i] = IntradayPattern.RANGE_EXPANSION
                continue

    return labels


def label_multi_session(
    df: pd.DataFrame,
    session_breaks: List[int] = None,
) -> pd.Series:
    """
    Label patterns across multiple trading sessions.

    Args:
        df: Multi-session 5-min OHLCV DataFrame with 'date' column.
        session_breaks: Optional list of indices where sessions start.
            If None, detected from date column gaps.

    Returns:
        Series of IntradayPattern int labels.
    """
    if session_breaks is None:
        session_breaks = _detect_session_breaks(df)

    all_labels = pd.Series(IntradayPattern.NO_PATTERN, index=df.index, dtype=int)
    prev_close = None

    for s_idx in range(len(session_breaks)):
        start = session_breaks[s_idx]
        end = session_breaks[s_idx + 1] if s_idx + 1 < len(session_breaks) else len(df)
        session_df = df.iloc[start:end]

        if len(session_df) < OPENING_RANGE_BARS + 2:
            prev_close = session_df["close"].iloc[-1] if len(session_df) > 0 else prev_close
            continue

        session_labels = label_session(session_df, prev_close=prev_close)
        all_labels.iloc[start:end] = session_labels.values
        prev_close = session_df["close"].iloc[-1]

    dist = all_labels.value_counts().to_dict()
    logger.info(f"Labeled {len(all_labels)} bars: {dict(dist)}")
    return all_labels


def _detect_session_breaks(df: pd.DataFrame) -> List[int]:
    """Detect session boundaries from timestamp gaps > 2 hours."""
    if "date" not in df.columns:
        # Fallback: assume continuous, split every BARS_PER_SESSION
        return list(range(0, len(df), BARS_PER_SESSION))

    dates = pd.to_datetime(df["date"])
    breaks = [0]
    for i in range(1, len(dates)):
        gap = (dates.iloc[i] - dates.iloc[i - 1]).total_seconds()
        if gap > 7200:  # > 2 hours = new session
            breaks.append(i)
    return breaks


def _fast_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute RSI using numpy for speed."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    rsi = np.full(len(close), np.nan)
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


def _fast_bb_pct_b(close: np.ndarray, period: int = 20) -> np.ndarray:
    """Compute Bollinger %B using numpy."""
    pct_b = np.full(len(close), np.nan)
    if len(close) < period:
        return pct_b

    for i in range(period - 1, len(close)):
        window = close[i - period + 1:i + 1]
        mid = np.mean(window)
        std = np.std(window, ddof=1)
        if std > 0:
            upper = mid + 2 * std
            lower = mid - 2 * std
            pct_b[i] = (close[i] - lower) / (upper - lower)

    return pct_b
