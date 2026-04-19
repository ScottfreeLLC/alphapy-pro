"""Triple Barrier Method labeling from de Prado's AFML."""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_daily_volatility(close: pd.Series, lookback: int = 20) -> pd.Series:
    """Compute daily volatility using exponentially weighted standard deviation of returns."""
    returns = close.pct_change()
    return returns.ewm(span=lookback).std()


def get_events(
    close: pd.Series,
    entries: pd.Series,
    vertical_barrier_periods: int = 10,
) -> pd.DataFrame:
    """
    Build events DataFrame from entry signals.

    Args:
        close: Price series with datetime index
        entries: Boolean series of entry signals
        vertical_barrier_periods: Maximum holding period in bars

    Returns:
        DataFrame with columns: t1 (vertical barrier timestamp), side (+1 long default)
    """
    entry_dates = entries[entries].index
    t1 = close.index.searchsorted(entry_dates) + vertical_barrier_periods
    t1 = np.clip(t1, 0, len(close.index) - 1)
    t1 = pd.Series(close.index[t1], index=entry_dates, name="t1")

    events = pd.DataFrame({"t1": t1, "side": 1})
    return events


def triple_barrier_labels(
    close: pd.Series,
    events: pd.DataFrame,
    pt_sl: tuple = (1.0, 1.0),
    min_ret: float = 0.0,
    volatility: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Apply Triple Barrier Method labeling.

    Three barriers:
    - Upper: take profit = pt_sl[0] * daily_vol
    - Lower: stop loss = pt_sl[1] * daily_vol
    - Vertical: time barrier (from events.t1)

    Args:
        close: Price series
        events: DataFrame with 't1' (vertical barrier) and 'side' columns
        pt_sl: (profit_taking_multiplier, stop_loss_multiplier) relative to volatility
        min_ret: Minimum return threshold for labeling as +1/-1
        volatility: Pre-computed volatility series; computed if None

    Returns:
        DataFrame with columns: ret (return), label (-1, 0, +1), t1 (exit time),
        barrier_touched ('tp', 'sl', 'vertical')
    """
    if volatility is None:
        volatility = get_daily_volatility(close)

    results = []

    for entry_time, row in events.iterrows():
        if entry_time not in close.index:
            continue

        t1 = row["t1"]
        side = row.get("side", 1)

        # Get the price path from entry to vertical barrier
        entry_idx = close.index.get_loc(entry_time)
        exit_idx = close.index.get_loc(t1) if t1 in close.index else len(close) - 1
        path = close.iloc[entry_idx : exit_idx + 1]

        if len(path) < 2:
            continue

        entry_price = path.iloc[0]
        vol = volatility.loc[entry_time] if entry_time in volatility.index else volatility.iloc[-1]

        if pd.isna(vol) or vol == 0:
            vol = 0.02  # fallback

        # Barrier levels
        upper = entry_price * (1 + pt_sl[0] * vol) if pt_sl[0] > 0 else np.inf
        lower = entry_price * (1 - pt_sl[1] * vol) if pt_sl[1] > 0 else -np.inf

        # Find first touch
        barrier_touched = "vertical"
        exit_time = t1
        exit_price = path.iloc[-1]

        for i in range(1, len(path)):
            price = path.iloc[i]
            if side == 1:  # Long
                if price >= upper:
                    barrier_touched = "tp"
                    exit_time = path.index[i]
                    exit_price = price
                    break
                elif price <= lower:
                    barrier_touched = "sl"
                    exit_time = path.index[i]
                    exit_price = price
                    break
            else:  # Short
                if price <= lower:
                    barrier_touched = "tp"
                    exit_time = path.index[i]
                    exit_price = price
                    break
                elif price >= upper:
                    barrier_touched = "sl"
                    exit_time = path.index[i]
                    exit_price = price
                    break

        ret = (exit_price - entry_price) / entry_price * side

        # Label based on return and barrier
        if barrier_touched == "tp":
            label = 1
        elif barrier_touched == "sl":
            label = -1
        else:
            label = 1 if ret > min_ret else (-1 if ret < -min_ret else 0)

        results.append({
            "entry_time": entry_time,
            "exit_time": exit_time,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "ret": round(ret, 6),
            "label": label,
            "barrier_touched": barrier_touched,
            "vol": round(vol, 6),
        })

    if not results:
        return pd.DataFrame(columns=["ret", "label", "barrier_touched", "vol", "exit_time"])

    df = pd.DataFrame(results).set_index("entry_time")
    logger.info(
        f"Triple barrier: {len(df)} labels — "
        f"+1={int((df['label'] == 1).sum())}, "
        f"-1={int((df['label'] == -1).sum())}, "
        f"0={int((df['label'] == 0).sum())}"
    )
    return df
