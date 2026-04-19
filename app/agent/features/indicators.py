"""Technical indicator computation using pandas-ta."""

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_trend(df: pd.DataFrame) -> Dict:
    """Compute trend indicators: SMA, EMA, price position."""
    close = df["close"]

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    ema20 = close.ewm(span=20, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    latest_close = close.iloc[-1]

    return {
        "sma20": _safe_round(sma20.iloc[-1]),
        "sma50": _safe_round(sma50.iloc[-1]),
        "ema20": _safe_round(ema20.iloc[-1]),
        "ema50": _safe_round(ema50.iloc[-1]),
        "price_vs_sma20_pct": _safe_round(_pct_diff(latest_close, sma20.iloc[-1])),
        "price_vs_sma50_pct": _safe_round(_pct_diff(latest_close, sma50.iloc[-1])),
        "sma20_slope": _safe_round(_slope(sma20)),
        "sma50_slope": _safe_round(_slope(sma50)),
        "above_sma20": bool(latest_close > sma20.iloc[-1]) if pd.notna(sma20.iloc[-1]) else None,
        "above_sma50": bool(latest_close > sma50.iloc[-1]) if pd.notna(sma50.iloc[-1]) else None,
        "golden_cross": bool(sma20.iloc[-1] > sma50.iloc[-1]) if pd.notna(sma50.iloc[-1]) else None,
    }


def compute_momentum(df: pd.DataFrame) -> Dict:
    """Compute momentum indicators: RSI, MACD, ROC."""
    close = df["close"]

    # RSI(14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # MACD(12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_histogram = macd_line - macd_signal

    # ROC(10)
    roc10 = close.pct_change(10) * 100

    return {
        "rsi_14": _safe_round(rsi.iloc[-1]),
        "rsi_zone": _rsi_zone(rsi.iloc[-1]),
        "macd_line": _safe_round(macd_line.iloc[-1]),
        "macd_signal": _safe_round(macd_signal.iloc[-1]),
        "macd_histogram": _safe_round(macd_histogram.iloc[-1]),
        "macd_bullish": bool(macd_line.iloc[-1] > macd_signal.iloc[-1]) if pd.notna(macd_signal.iloc[-1]) else None,
        "roc_10": _safe_round(roc10.iloc[-1]),
    }


def compute_volatility(df: pd.DataFrame) -> Dict:
    """Compute volatility indicators: Bollinger Bands, ATR, historical volatility."""
    close = df["close"]
    high = df["high"]
    low = df["low"]

    # Bollinger Bands(20, 2)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_width = ((bb_upper - bb_lower) / bb_mid * 100) if pd.notna(bb_mid.iloc[-1]) and bb_mid.iloc[-1] != 0 else None
    bb_pct_b = (close - bb_lower) / (bb_upper - bb_lower)

    # ATR(14)
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    # 20-day historical volatility (annualized)
    daily_returns = close.pct_change()
    hist_vol = daily_returns.rolling(20).std() * np.sqrt(252) * 100

    return {
        "bb_upper": _safe_round(bb_upper.iloc[-1]),
        "bb_middle": _safe_round(bb_mid.iloc[-1]),
        "bb_lower": _safe_round(bb_lower.iloc[-1]),
        "bb_width_pct": _safe_round(bb_width.iloc[-1]) if isinstance(bb_width, pd.Series) else _safe_round(bb_width),
        "bb_pct_b": _safe_round(bb_pct_b.iloc[-1]),
        "bb_position": _bb_position(bb_pct_b.iloc[-1]),
        "atr_14": _safe_round(atr.iloc[-1]),
        "atr_pct": _safe_round(atr.iloc[-1] / close.iloc[-1] * 100) if close.iloc[-1] != 0 else None,
        "hist_volatility_20d": _safe_round(hist_vol.iloc[-1]),
    }


def compute_volume(df: pd.DataFrame) -> Dict:
    """Compute volume indicators: volume SMA, ratio, OBV, trend."""
    close = df["close"]
    volume = df["volume"]

    vol_sma20 = volume.rolling(20).mean()
    vol_ratio = volume.iloc[-1] / vol_sma20.iloc[-1] if pd.notna(vol_sma20.iloc[-1]) and vol_sma20.iloc[-1] > 0 else None

    # OBV
    obv = (np.sign(close.diff()) * volume).cumsum()

    # Volume trend (5-day vs 20-day average)
    vol_sma5 = volume.rolling(5).mean()
    vol_trend = "increasing" if pd.notna(vol_sma5.iloc[-1]) and pd.notna(vol_sma20.iloc[-1]) and vol_sma5.iloc[-1] > vol_sma20.iloc[-1] else "decreasing"

    return {
        "volume_sma20": _safe_round(vol_sma20.iloc[-1], 0),
        "volume_ratio": _safe_round(vol_ratio),
        "volume_surge": bool(vol_ratio > 1.5) if vol_ratio is not None else False,
        "obv": _safe_round(obv.iloc[-1], 0),
        "obv_trend": "up" if len(obv) >= 5 and obv.iloc[-1] > obv.iloc[-5] else "down",
        "volume_trend": vol_trend,
    }


def compute_vwap(df: pd.DataFrame) -> Dict:
    """Compute VWAP and related metrics for intraday bars."""
    if "vwap" in df.columns:
        # Use pre-computed VWAP from Massive if available
        vwap = df["vwap"]
    else:
        # Compute cumulative VWAP from OHLCV
        typical_price = (df["high"] + df["low"] + df["close"]) / 3
        cum_tp_vol = (typical_price * df["volume"]).cumsum()
        cum_vol = df["volume"].cumsum()
        vwap = cum_tp_vol / cum_vol.replace(0, np.nan)

    latest_close = df["close"].iloc[-1]
    latest_vwap = vwap.iloc[-1]

    # VWAP distance
    vwap_dist = (latest_close - latest_vwap) / latest_vwap * 100 if pd.notna(latest_vwap) and latest_vwap != 0 else None

    # VWAP slope (rising or falling over last 5 bars)
    vwap_slope = _slope(vwap, lookback=5) if len(vwap) >= 5 else 0.0

    return {
        "vwap": _safe_round(latest_vwap),
        "vwap_distance_pct": _safe_round(vwap_dist),
        "above_vwap": bool(latest_close > latest_vwap) if pd.notna(latest_vwap) else None,
        "vwap_slope": _safe_round(vwap_slope),
    }


def compute_price_context(df: pd.DataFrame) -> Dict:
    """Compute price context: distance from highs/lows, daily change."""
    close = df["close"]
    high = df["high"]
    low = df["low"]

    high_20d = high.rolling(20).max()
    low_20d = low.rolling(20).min()

    latest_close = close.iloc[-1]
    prev_close = close.iloc[-2] if len(close) >= 2 else latest_close

    daily_change = latest_close - prev_close
    daily_change_pct = (daily_change / prev_close * 100) if prev_close != 0 else 0

    dist_from_high = ((high_20d.iloc[-1] - latest_close) / high_20d.iloc[-1] * 100) if pd.notna(high_20d.iloc[-1]) and high_20d.iloc[-1] != 0 else 0
    dist_from_low = ((latest_close - low_20d.iloc[-1]) / low_20d.iloc[-1] * 100) if pd.notna(low_20d.iloc[-1]) and low_20d.iloc[-1] != 0 else 0

    return {
        "daily_change": _safe_round(daily_change),
        "daily_change_pct": _safe_round(daily_change_pct),
        "high_20d": _safe_round(high_20d.iloc[-1]),
        "low_20d": _safe_round(low_20d.iloc[-1]),
        "dist_from_20d_high_pct": _safe_round(dist_from_high),
        "dist_from_20d_low_pct": _safe_round(dist_from_low),
        "near_20d_high": bool(dist_from_high < 2.0),
        "near_20d_low": bool(dist_from_low < 2.0),
    }


def build_trend_summary(
    symbol: str,
    price: float,
    trend: Dict,
    momentum: Dict,
    volatility: Dict,
    volume: Dict,
    price_ctx: Dict,
) -> str:
    """Build a human-readable trend summary for the LLM."""
    parts = []

    # Price and trend
    parts.append(f"{symbol} at ${price:.2f}.")

    if trend.get("above_sma20") and trend.get("above_sma50"):
        parts.append("Strong uptrend: price above both SMA20 and SMA50.")
    elif trend.get("above_sma20"):
        parts.append("Short-term uptrend (above SMA20), but below SMA50.")
    elif trend.get("above_sma50"):
        parts.append("Above SMA50 but pulled back below SMA20.")
    else:
        parts.append("Downtrend: below both SMA20 and SMA50.")

    if trend.get("golden_cross"):
        parts.append("SMA20 > SMA50 (golden cross).")
    elif trend.get("golden_cross") is False:
        parts.append("SMA20 < SMA50 (death cross).")

    # Momentum
    rsi = momentum.get("rsi_14")
    if rsi is not None:
        zone = momentum.get("rsi_zone", "neutral")
        parts.append(f"RSI(14) = {rsi:.1f} ({zone}).")

    if momentum.get("macd_bullish"):
        parts.append("MACD is bullish (above signal line).")
    elif momentum.get("macd_bullish") is False:
        parts.append("MACD is bearish (below signal line).")

    roc = momentum.get("roc_10")
    if roc is not None:
        direction = "up" if roc > 0 else "down"
        parts.append(f"10-day ROC: {roc:+.1f}% ({direction}).")

    # Volatility
    bb_pos = volatility.get("bb_position")
    if bb_pos:
        parts.append(f"Bollinger Band position: {bb_pos}.")

    atr_pct = volatility.get("atr_pct")
    if atr_pct is not None:
        parts.append(f"ATR is {atr_pct:.1f}% of price.")

    # Volume
    vol_ratio = volume.get("volume_ratio")
    if vol_ratio is not None:
        if volume.get("volume_surge"):
            parts.append(f"Volume surge: {vol_ratio:.1f}x average (significant).")
        else:
            parts.append(f"Volume ratio: {vol_ratio:.1f}x average.")

    parts.append(f"Volume trend: {volume.get('volume_trend', 'unknown')}.")

    # Price context
    if price_ctx.get("near_20d_high"):
        parts.append("Near 20-day high.")
    elif price_ctx.get("near_20d_low"):
        parts.append("Near 20-day low.")

    change_pct = price_ctx.get("daily_change_pct", 0)
    if change_pct != 0:
        parts.append(f"Daily change: {change_pct:+.2f}%.")

    return " ".join(parts)


# --- Helpers ---

def _safe_round(value, decimals=2):
    """Safely round a value, handling NaN/None."""
    if value is None or (isinstance(value, float) and (np.isnan(value) or np.isinf(value))):
        return None
    return round(float(value), decimals)


def _pct_diff(a, b):
    """Percentage difference of a relative to b."""
    if b is None or b == 0 or (isinstance(b, float) and np.isnan(b)):
        return None
    return (a - b) / b * 100


def _slope(series: pd.Series, lookback: int = 5) -> float:
    """Simple slope of last N values (positive = rising)."""
    if len(series) < lookback:
        return 0.0
    recent = series.iloc[-lookback:]
    if recent.isna().any():
        return 0.0
    return float(recent.iloc[-1] - recent.iloc[0])


def _rsi_zone(rsi_value) -> str:
    """Classify RSI into zones."""
    if rsi_value is None or (isinstance(rsi_value, float) and np.isnan(rsi_value)):
        return "unknown"
    if rsi_value >= 70:
        return "overbought"
    if rsi_value >= 60:
        return "bullish"
    if rsi_value >= 40:
        return "neutral"
    if rsi_value >= 30:
        return "bearish"
    return "oversold"


def _bb_position(pct_b) -> str:
    """Classify Bollinger Band %B position."""
    if pct_b is None or (isinstance(pct_b, float) and np.isnan(pct_b)):
        return "unknown"
    if pct_b >= 1.0:
        return "above upper band"
    if pct_b >= 0.8:
        return "near upper band"
    if pct_b >= 0.2:
        return "middle of bands"
    if pct_b >= 0.0:
        return "near lower band"
    return "below lower band"
