---
name: crypto_momentum
timeframes: [1h, 4h]
risk_per_trade: 0.01
max_positions: 2
enabled: true
tags: [crypto, momentum, trend-following]
---

# Crypto Momentum Strategy

## Entry Conditions (ALL must be met)

1. **Trending market**: Price is above both the 20-period and 50-period moving averages on the evaluation timeframe, confirming a strong uptrend.

2. **Momentum acceleration**: The rate of price change is increasing — the last 4 bars show progressively higher closes (or at least 3 of 4 bars are green).

3. **Volume confirmation**: Volume on up-bars exceeds volume on down-bars over the last 10 periods, showing buying pressure dominance.

4. **Not overextended**: Price is not more than 8% above the 20-period SMA. Avoid chasing extended moves.

## Short Entry Conditions

Mirror the above for shorts:
- Price below both 20 and 50 period MAs
- Momentum decelerating (progressively lower closes)
- Volume on down-bars exceeds up-bars
- Price not more than 8% below 20-period SMA

## Exit Conditions

- **Stop loss**: 3% from entry (crypto is more volatile than stocks, needs wider stops).
- **Take profit**: 6% from entry (2:1 risk/reward minimum).
- **Trailing stop**: Once in profit by 4%, trail stop at 2% from the high.

## Position Sizing

- Risk 1% of account per trade (reduced due to crypto volatility)
- Maximum 2 concurrent crypto positions

## Filters

- Only trade BTC, ETH, SOL (high-liquidity cryptos)
- Avoid trading during known high-impact events (Fed announcements, etc.)
- Reduce position size by 50% on weekends (lower liquidity)
