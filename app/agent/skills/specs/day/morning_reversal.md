---
name: morning_reversal
timeframes: [5min]
risk_per_trade: 0.01
max_positions: 2
enabled: true
tags: [intraday, reversal, mean-reversion]
---

# Morning Reversal Strategy

## Entry Conditions (ALL must be met)

1. **Large gap**: Stock has gapped at least 2% from previous close (up or down).

2. **Failed continuation**: After the gap, price moves further in the gap direction in the first 15 minutes, then reverses. For gap-up: price makes a new high in first 15 min, then drops below the open. For gap-down: price makes a new low, then rises above the open.

3. **Reversal confirmation**: A 5-min bar closes in the reversal direction with above-average volume, indicating the gap is being faded.

4. **RSI extreme**: 5-min RSI(14) was above 75 (for gap-up fade) or below 25 (for gap-down fade) before the reversal bar.

5. **Time window**: Entry occurs between 9:45–11:00 ET (morning reversal window).

## Exit Conditions

- **Stop loss**: Place stop at the intraday extreme (high of day for short, low of day for long).
- **Take profit**: Target the previous day's close (gap fill) as the primary target.
- **Partial exit**: Close half the position at 50% of the gap fill level.
- **Time stop**: Close remaining position by 14:00 ET if target not reached.

## Position Sizing

- Risk 1% of account per trade
- Maximum 2 concurrent reversal positions

## Filters

- Minimum gap size: 2% from previous close
- Maximum gap size: 8% (larger gaps may have fundamental catalysts)
- Minimum average daily volume: 1,000,000 shares
- Skip if there's a known earnings or FDA catalyst for the gap
