---
name: vwap_reclaim
timeframes: [5min]
risk_per_trade: 0.01
max_positions: 2
enabled: true
tags: [intraday, vwap, mean-reversion]
---

# VWAP Reclaim Strategy

## Entry Conditions (ALL must be met)

1. **Below VWAP setup**: Price has been trading below VWAP for at least 30 minutes (6+ bars on 5-min chart).

2. **VWAP reclaim**: A 5-min bar closes decisively above VWAP (close is at least 0.1% above VWAP, not just touching).

3. **Volume surge**: The reclaim bar has volume at least 1.3x the session average volume up to that point.

4. **Trend context**: The stock is in a broader uptrend (above daily 20-SMA) — this filters for stocks that are temporarily weak, not structurally bearish.

5. **Time window**: Entry between 10:00–14:30 ET (avoids opening noise and closing imbalances).

## Exit Conditions

- **Stop loss**: Place stop at the session low or 0.5% below VWAP, whichever is tighter.
- **Take profit**: Target the session high or 1.5x risk distance above entry.
- **Time stop**: Close by 15:45 ET — no overnight holds.
- **Trailing stop**: Once price is 1x risk above entry, trail stop to VWAP.

## Position Sizing

- Risk 1% of account per trade
- Maximum 2 concurrent VWAP reclaim positions

## Filters

- Only for stocks currently above their daily 20-SMA (uptrend context)
- Minimum average daily volume: 500,000 shares
- Skip if stock has been below VWAP for less than 30 minutes (too early)
