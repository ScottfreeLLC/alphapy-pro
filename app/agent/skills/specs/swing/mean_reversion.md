---
name: mean_reversion
timeframes: [1d, 4h]
risk_per_trade: 0.015
max_positions: 3
enabled: true
tags: [mean-reversion, oversold, bounce]
---

# Mean Reversion Strategy

## Entry Conditions (ALL must be met)

1. **Oversold condition**: Price has pulled back at least 5% from its 20-day high, indicating a potential oversold bounce opportunity.

2. **Support confluence**: Price is near a recognized pivot low or support level from the pattern analysis (within 1% of the level).

3. **Bullish reversal candle**: The most recent bar shows a bullish signal — close above open with the low testing support.

4. **Trend context**: The stock's 50-day trend is still positive (price above 50-period SMA). We're buying a dip in an uptrend, not catching a falling knife.

## Exit Conditions

- **Stop loss**: Place stop 1% below the support level or pivot low being tested.
- **Take profit**: Target the 20-day moving average or the most recent pivot high, whichever is closer.
- **Time stop**: If the position hasn't reached 50% of target within 5 trading days, exit at market.

## Position Sizing

- Risk 1.5% of account per trade
- Maximum 3 concurrent positions from this strategy

## Filters

- Only trade stocks with positive 50-day momentum
- Skip if daily RSI is below 20 (extreme oversold — wait for confirmation)
- Skip if the pullback is news-driven (earnings miss, downgrade)
