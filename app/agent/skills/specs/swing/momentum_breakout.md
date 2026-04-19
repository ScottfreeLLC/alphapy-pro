---
name: momentum_breakout
timeframes: [1d, 1h]
risk_per_trade: 0.02
max_positions: 3
enabled: true
tags: [momentum, breakout, trend-following]
---

# Momentum Breakout Strategy

## Entry Conditions (ALL must be met)

1. **Price above 20-day SMA**: The current price must be trading above its 20-period simple moving average, confirming an uptrend.

2. **Volume surge**: Current volume is at least 1.5x the 20-period average volume, indicating institutional participation.

3. **Breakout above resistance**: Price has broken above the most recent pivot high (resistance level) within the last 3 bars.

4. **Relative strength**: The stock's daily change is outperforming the broader market (positive daily change when market is flat or positive).

## Exit Conditions

- **Stop loss**: Place stop at the most recent pivot low or 2% below entry, whichever is tighter.
- **Take profit**: Target 2x the risk (distance from entry to stop). If risk is $2, target is $4 above entry.
- **Trailing stop**: Once in profit by 1x risk, trail stop to breakeven. Once in profit by 1.5x risk, trail stop to 1x risk.

## Position Sizing

- Risk 2% of account per trade
- Maximum 3 concurrent positions from this strategy

## Filters

- Skip stocks with average daily volume below 500,000 shares
- Skip stocks priced below $10
- Avoid earnings week (if known)
