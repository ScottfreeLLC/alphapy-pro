"""
Strategy signal generators for backtesting.

Each function takes a pandas DataFrame (OHLCV columns: open, high, low, close, volume)
and returns a dict with:
  - entries: boolean pd.Series of entry signals
  - sl_stop: float stop-loss percentage (e.g. 0.02 = 2%)
  - tp_stop: float take-profit percentage
"""

import pandas as pd


def momentum_breakout_signals(df: pd.DataFrame) -> dict:
    """Momentum breakout — maps to momentum_breakout.md skill.

    Entry when price breaks above resistance on high volume in an uptrend.
    """
    close = df["close"]
    volume = df["volume"]

    sma20 = close.rolling(20).mean()
    vol_avg = volume.rolling(20).mean()
    resistance = close.rolling(10).max().shift(1)

    entries = (
        (close > sma20)
        & (volume > vol_avg * 1.5)
        & (close > resistance)
        & (close.pct_change() > 0)
        & (vol_avg > 500_000)
        & (close > 10)
    )

    return {"entries": entries.fillna(False), "sl_stop": 0.02, "tp_stop": 0.04}


def mean_reversion_signals(df: pd.DataFrame) -> dict:
    """Mean reversion — maps to mean_reversion.md skill.

    Entry on oversold bounce near support in an overall uptrend.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]

    high_20d = close.rolling(20).max()
    pullback_pct = (high_20d - close) / high_20d
    sma50 = close.rolling(50).mean()
    recent_low = low.rolling(20).min()
    near_support = (close - recent_low) / close

    entries = (
        (pullback_pct >= 0.05)
        & (near_support <= 0.01)
        & (close > open_)
        & (close > sma50)
    )

    return {"entries": entries.fillna(False), "sl_stop": 0.01, "tp_stop": 0.03}


def crypto_momentum_signals(df: pd.DataFrame) -> dict:
    """Crypto momentum — maps to crypto_momentum.md skill.

    Entry when price is trending above both SMAs with momentum acceleration
    and not overextended.
    """
    close = df["close"]

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    green_bars = (close > df["open"]).rolling(4).sum()
    extension = (close - sma20) / sma20

    entries = (
        (close > sma20)
        & (close > sma50)
        & (green_bars >= 3)
        & (extension < 0.08)
        & (extension > 0)
    )

    return {"entries": entries.fillna(False), "sl_stop": 0.03, "tp_stop": 0.06}


def pivot_pattern_signals(df: pd.DataFrame) -> dict:
    """Pivot pattern entry — maps to pivot_pattern_entry.md skill.

    Simplified version: entry when multiple bullish reversal conditions align
    (higher lows, bullish candle, volume confirmation).
    """
    close = df["close"]
    open_ = df["open"]
    low = df["low"]
    volume = df["volume"]

    # Detect higher lows (simple proxy for bullish pivot pattern)
    low_5 = low.rolling(5).min()
    prev_low_5 = low_5.shift(5)
    higher_lows = low_5 > prev_low_5

    # Bullish candle
    bullish_bar = close > open_

    # Volume above average
    vol_avg = volume.rolling(20).mean()
    vol_confirm = volume > vol_avg

    # Multiple conditions acting as "pattern confluence"
    sma20 = close.rolling(20).mean()
    above_sma = close > sma20

    entries = higher_lows & bullish_bar & vol_confirm & above_sma

    return {"entries": entries.fillna(False), "sl_stop": 0.02, "tp_stop": 0.04}


def ml_meta_label_signals(df: pd.DataFrame) -> dict:
    """ML meta-label strategy — combines momentum breakout with ML filtering.

    Uses the same entry conditions as momentum_breakout but is designed
    to be paired with the MetaModel for signal filtering in ML backtests.
    """
    close = df["close"]
    volume = df["volume"]

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    vol_avg = volume.rolling(20).mean()
    resistance = close.rolling(10).max().shift(1)

    # Broader entry criteria than momentum_breakout to generate more signals for ML training
    entries = (
        (close > sma20)
        & (close > sma50)
        & (volume > vol_avg * 1.2)
        & (close > resistance)
    )

    return {"entries": entries.fillna(False), "sl_stop": 0.02, "tp_stop": 0.04}


# ============================================================================
# Intraday strategies (5-min bars)
# ============================================================================

def _compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Compute VWAP for intraday bars (resets at session boundary)."""
    typical = (df["high"] + df["low"] + df["close"]) / 3
    cum_tp_vol = (typical * df["volume"]).cumsum()
    cum_vol = df["volume"].cumsum()
    return (cum_tp_vol / cum_vol.replace(0, float("nan"))).ffill()


def _bar_index(df: pd.DataFrame) -> pd.Series:
    """Assign bar index 0-77 within each trading day (78 five-min bars per session)."""
    if hasattr(df.index, "hour"):
        # Group by date, enumerate within each group
        dates = df.index.date
        idx = df.groupby(dates).cumcount()
        return idx
    return pd.Series(range(len(df)), index=df.index)


def _opening_range(df: pd.DataFrame, n_bars: int = 6) -> tuple:
    """Compute opening range (first n_bars, default 30 min = 6 five-min bars).

    Returns (or_high, or_low) as Series aligned to the full DataFrame.
    """
    dates = df.index.date if hasattr(df.index, "date") else pd.Series(range(len(df)), index=df.index)
    bar_idx = df.groupby(dates).cumcount()

    # Mark the opening range window
    or_mask = bar_idx < n_bars

    # Compute OR high/low per day, forward-fill across the full session
    or_highs = df["high"].where(or_mask).groupby(dates).transform("max")
    or_lows = df["low"].where(or_mask).groupby(dates).transform("min")

    return or_highs, or_lows


def day_orb_breakout_signals(df: pd.DataFrame) -> dict:
    """Opening Range Breakout — maps to orb_breakout.md skill.

    Entry when price breaks above the 30-min opening range with volume confirmation,
    after the opening range period completes (bar 6+).
    """
    close = df["close"]
    volume = df["volume"]

    or_high, or_low = _opening_range(df)
    bar_idx = _bar_index(df)
    vol_avg = volume.rolling(20).mean()

    entries = (
        (bar_idx >= 6)           # After opening range completes
        & (close > or_high)      # Break above OR high
        & (volume > vol_avg * 1.3)  # Volume surge
        & (close > close.shift(1))  # Current bar is green
    )

    return {"entries": entries.fillna(False), "sl_stop": 0.005, "tp_stop": 0.01}


def day_morning_reversal_signals(df: pd.DataFrame) -> dict:
    """Morning Reversal — maps to morning_reversal.md skill.

    Entry on reversal from opening range low, with bullish candle and volume,
    during first 2 hours of trading (bars 6-30).
    """
    close = df["close"]
    open_ = df["open"]
    low = df["low"]
    volume = df["volume"]

    or_high, or_low = _opening_range(df)
    bar_idx = _bar_index(df)
    vol_avg = volume.rolling(20).mean()

    # Touched or breached OR low, then reversed up
    near_or_low = (low <= or_low * 1.002)
    bullish_candle = close > open_
    recovering = close > or_low

    entries = (
        (bar_idx >= 6)
        & (bar_idx <= 30)          # First 2.5 hours
        & near_or_low
        & bullish_candle
        & recovering
        & (volume > vol_avg * 1.2)
    )

    return {"entries": entries.fillna(False), "sl_stop": 0.004, "tp_stop": 0.008}


def day_vwap_reclaim_signals(df: pd.DataFrame) -> dict:
    """VWAP Reclaim — maps to vwap_reclaim.md skill.

    Entry when price crosses back above VWAP after trading below it,
    with volume confirmation.
    """
    close = df["close"]
    volume = df["volume"]

    vwap = _compute_vwap(df)
    vol_avg = volume.rolling(20).mean()

    # Was below VWAP, now reclaiming
    below_vwap_prev = close.shift(1) < vwap.shift(1)
    above_vwap_now = close > vwap
    bar_idx = _bar_index(df)

    entries = (
        below_vwap_prev
        & above_vwap_now
        & (volume > vol_avg * 1.2)
        & (bar_idx >= 3)          # Skip first 15 min noise
        & (bar_idx <= 65)         # Not in last 65 min
    )

    return {"entries": entries.fillna(False), "sl_stop": 0.004, "tp_stop": 0.008}


def day_gap_fill_signals(df: pd.DataFrame) -> dict:
    """Gap Fill — maps to gap_fill.md skill.

    Entry when a gap-down stock starts filling the gap back toward
    the previous close. Requires gap >= 1% and price recovering.
    """
    close = df["close"]
    open_ = df["open"]
    volume = df["volume"]

    bar_idx = _bar_index(df)
    vol_avg = volume.rolling(20).mean()

    # Identify the session open (bar 0) close for gap reference
    dates = df.index.date if hasattr(df.index, "date") else pd.Series(range(len(df)), index=df.index)
    first_bar_open = open_.groupby(dates).transform("first")

    # Previous session last close (shift by bar index)
    prev_close = close.shift(1).groupby(dates).transform("first")

    # Gap down: session opened > 1% below previous close
    gap_pct = (first_bar_open - prev_close) / prev_close
    is_gap_down = gap_pct < -0.01

    # Price recovering toward previous close
    filling = close > first_bar_open
    bullish = close > open_

    entries = (
        is_gap_down
        & filling
        & bullish
        & (bar_idx >= 3)
        & (bar_idx <= 40)         # Gap fills typically happen in first half
        & (volume > vol_avg * 1.1)
    )

    return {"entries": entries.fillna(False), "sl_stop": 0.005, "tp_stop": 0.01}


def day_power_hour_signals(df: pd.DataFrame) -> dict:
    """Power Hour — maps to power_hour.md skill.

    Entry during the final hour of trading (3:00-4:00 PM ET, bars 66-77)
    on momentum acceleration with volume surge.
    """
    close = df["close"]
    volume = df["volume"]

    bar_idx = _bar_index(df)
    vol_avg = volume.rolling(20).mean()
    sma10 = close.rolling(10).mean()

    # Momentum: price above short-term MA and rising
    momentum = (close > sma10) & (close > close.shift(1)) & (close.shift(1) > close.shift(2))

    entries = (
        (bar_idx >= 66)          # Last hour (3:00-4:00 PM)
        & momentum
        & (volume > vol_avg * 1.5)  # Strong volume in power hour
    )

    return {"entries": entries.fillna(False), "sl_stop": 0.003, "tp_stop": 0.006}


# ============================================================================
# Encoded pattern strategies
# ============================================================================

def encoded_pattern_signals(df: pd.DataFrame) -> dict:
    """Encoded pattern entry — uses price encoding to detect pivot/net patterns.

    Entry when last 5 bars match a bullish reversal pattern from the encoding library.
    """
    from ..ml.encoding.encoder import encode_price_df
    from ..ml.encoding.patterns import find_patterns

    enc_df = encode_price_df(df, p=20)
    encoded_tokens = " ".join(enc_df["encoded_str"].values)

    entries = pd.Series(False, index=df.index)

    # Scan with a sliding 5-bar window for bullish patterns
    for i in range(4, len(df)):
        window_tokens = " ".join(enc_df["encoded_str"].iloc[i - 4 : i + 1].values)
        matches = find_patterns(window_tokens, pattern_types=["bullish"])
        if matches:
            entries.iloc[i] = True

    return {"entries": entries, "sl_stop": 0.02, "tp_stop": 0.04}


def day_encoded_pattern_signals(df: pd.DataFrame) -> dict:
    """Intraday encoded pattern entry — bullish patterns on 5-min bars.

    Same logic as swing but with tighter stops for intraday.
    """
    from ..ml.encoding.encoder import encode_price_df
    from ..ml.encoding.patterns import find_patterns

    enc_df = encode_price_df(df, p=20)

    entries = pd.Series(False, index=df.index)

    for i in range(4, len(df)):
        window_tokens = " ".join(enc_df["encoded_str"].iloc[i - 4 : i + 1].values)
        matches = find_patterns(window_tokens, pattern_types=["bullish"])
        if matches:
            entries.iloc[i] = True

    return {"entries": entries, "sl_stop": 0.005, "tp_stop": 0.01}


# ============================================================================
# Strategy registries
# ============================================================================

SWING_STRATEGIES = {
    "momentum_breakout": momentum_breakout_signals,
    "mean_reversion": mean_reversion_signals,
    "crypto_momentum": crypto_momentum_signals,
    "pivot_pattern_entry": pivot_pattern_signals,
    "ml_meta_label": ml_meta_label_signals,
    "encoded_pattern": encoded_pattern_signals,
}

DAY_STRATEGIES = {
    "day_orb_breakout": day_orb_breakout_signals,
    "day_morning_reversal": day_morning_reversal_signals,
    "day_vwap_reclaim": day_vwap_reclaim_signals,
    "day_gap_fill": day_gap_fill_signals,
    "day_power_hour": day_power_hour_signals,
    "day_encoded_pattern": day_encoded_pattern_signals,
}

# Combined registry (backward-compatible)
STRATEGIES = {**SWING_STRATEGIES, **DAY_STRATEGIES}
