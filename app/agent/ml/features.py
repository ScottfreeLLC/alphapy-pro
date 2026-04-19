"""ML feature pipeline: convert OHLCV to ML-ready feature matrix."""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from ..features.engine import FeatureEngine

logger = logging.getLogger(__name__)


def build_feature_matrix(
    df: pd.DataFrame,
    extra_dfs: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Build ML-ready feature matrix from OHLCV DataFrame.

    Features:
    - Phase 1 technical indicators (SMA, RSI, MACD, BB, ATR, volume)
    - Fractionally differentiated close prices
    - Multi-window realized volatility (5/10/20/60)
    - Rolling Sharpe and Sortino ratios
    - Volume imbalance
    - Cross-asset features (BTC, VIX proxy) if provided

    Args:
        df: OHLCV DataFrame with columns: open, high, low, close, volume
        extra_dfs: Optional dict of {"BTC": btc_df, "VIX": vix_df} for cross-asset features

    Returns:
        DataFrame with one row per bar, columns are features
    """
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    returns = close.pct_change()

    features = pd.DataFrame(index=df.index)

    # --- Phase 1 indicators ---
    features["sma20"] = close.rolling(20).mean()
    features["sma50"] = close.rolling(50).mean()
    features["ema20"] = close.ewm(span=20, adjust=False).mean()
    features["price_vs_sma20"] = (close - features["sma20"]) / features["sma20"]
    features["price_vs_sma50"] = (close - features["sma50"]) / features["sma50"]
    features["sma20_above_sma50"] = (features["sma20"] > features["sma50"]).astype(int)

    # RSI(14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    features["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    features["macd"] = ema12 - ema26
    features["macd_signal"] = features["macd"].ewm(span=9, adjust=False).mean()
    features["macd_hist"] = features["macd"] - features["macd_signal"]

    # Bollinger Bands
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    features["bb_pct_b"] = (close - (bb_mid - 2 * bb_std)) / (4 * bb_std)
    features["bb_width"] = (4 * bb_std) / bb_mid

    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    features["atr_14"] = tr.rolling(14).mean()
    features["atr_pct"] = features["atr_14"] / close

    # Volume
    vol_sma20 = volume.rolling(20).mean()
    features["volume_ratio"] = volume / vol_sma20
    features["obv"] = (np.sign(close.diff()) * volume).cumsum()
    features["obv_pct_change"] = features["obv"].pct_change(5)

    # ROC
    features["roc_5"] = close.pct_change(5)
    features["roc_10"] = close.pct_change(10)
    features["roc_20"] = close.pct_change(20)

    # --- Fractionally differentiated features (simplified via expanding window) ---
    features["frac_diff_close"] = _frac_diff(close, d=0.4)

    # --- Multi-window realized volatility ---
    for window in [5, 10, 20, 60]:
        features[f"realized_vol_{window}"] = returns.rolling(window).std() * np.sqrt(252)

    # --- Rolling Sharpe and Sortino ---
    features["rolling_sharpe_20"] = _rolling_sharpe(returns, 20)
    features["rolling_sortino_20"] = _rolling_sortino(returns, 20)

    # --- Volume imbalance ---
    up_vol = volume.where(close.diff() > 0, 0)
    down_vol = volume.where(close.diff() <= 0, 0)
    features["volume_imbalance"] = (
        up_vol.rolling(10).sum() - down_vol.rolling(10).sum()
    ) / volume.rolling(10).sum()

    # --- Price context ---
    features["dist_from_20d_high"] = (high.rolling(20).max() - close) / close
    features["dist_from_20d_low"] = (close - low.rolling(20).min()) / close
    features["daily_return"] = returns

    # --- Cross-asset features ---
    if extra_dfs:
        for asset_name, asset_df in extra_dfs.items():
            if asset_df is not None and "close" in asset_df.columns:
                asset_close = asset_df["close"].reindex(df.index, method="ffill")
                features[f"{asset_name.lower()}_return_5d"] = asset_close.pct_change(5)
                features[f"{asset_name.lower()}_vol_20d"] = (
                    asset_close.pct_change().rolling(20).std() * np.sqrt(252)
                )

    # --- Price encoding features ---
    try:
        from .encoding.encoder import encode_price_df
        enc_df = encode_price_df(df, p=20)
        features["pivot_strength"] = enc_df["pivot_strength"]
        features["pivot_direction"] = enc_df["pivot_direction"]
        features["net_direction"] = enc_df["net_direction"]
        features["net_magnitude"] = enc_df["net_magnitude"]
        features["range_magnitude"] = enc_df["range_magnitude"]
        features["volume_magnitude"] = enc_df["volume_magnitude"]
    except Exception as e:
        logger.warning(f"Price encoding features skipped: {e}")

    # Drop initial NaN rows and replace remaining NaN with 0
    features = features.iloc[60:]  # Skip warmup period
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)

    logger.info(f"Feature matrix: {features.shape[0]} rows x {features.shape[1]} columns")
    return features


def _frac_diff(series: pd.Series, d: float = 0.4, threshold: float = 1e-5) -> pd.Series:
    """
    Fractionally differentiated series (de Prado Ch. 5).

    Uses a fixed-width window approach for efficiency.
    """
    weights = [1.0]
    for k in range(1, len(series)):
        w = -weights[-1] * (d - k + 1) / k
        if abs(w) < threshold:
            break
        weights.append(w)

    weights = np.array(weights[::-1])
    width = len(weights)

    result = pd.Series(index=series.index, dtype=float)
    for i in range(width - 1, len(series)):
        window = series.iloc[i - width + 1 : i + 1].values
        if len(window) == width:
            result.iloc[i] = np.dot(weights, window)

    return result


def _rolling_sharpe(returns: pd.Series, window: int) -> pd.Series:
    """Rolling annualized Sharpe ratio."""
    mean = returns.rolling(window).mean()
    std = returns.rolling(window).std()
    return (mean / std.replace(0, np.nan)) * np.sqrt(252)


def _rolling_sortino(returns: pd.Series, window: int) -> pd.Series:
    """Rolling annualized Sortino ratio."""
    mean = returns.rolling(window).mean()
    downside = returns.where(returns < 0, 0)
    downside_std = downside.rolling(window).std()
    return (mean / downside_std.replace(0, np.nan)) * np.sqrt(252)
