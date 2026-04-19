---
name: orb_breakout
timeframes: [5min]
risk_per_trade: 0.01
max_positions: 2
enabled: true
tags: [intraday, breakout, opening-range]
---

# Opening Range Breakout (ORB) Strategy

## Entry Conditions (ALL must be met)

1. **Opening range defined**: The high and low of the first 30 minutes (9:30–10:00 ET) establish the opening range.

2. **Breakout confirmation**: Price closes a 5-min bar above the opening range high (long) or below the opening range low (short) after 10:00 ET.

3. **Volume confirmation**: Breakout bar volume is at least 1.5x the average volume of the opening range bars.

4. **Gap alignment**: For longs, the stock gapped up or opened flat. For shorts, the stock gapped down or opened flat. Avoid fading large gaps.

5. **VWAP alignment**: For longs, price is above VWAP at breakout. For shorts, price is below VWAP.

## Exit Conditions

- **Stop loss**: Place stop at the opposite end of the opening range (for longs, stop at OR low; for shorts, stop at OR high).
- **Take profit**: Target 1.5x the opening range width from the breakout level.
- **Time stop**: Close any remaining position by 15:30 ET — no overnight holds.
- **Trailing stop**: Once in profit by 1x risk, trail stop to breakeven.

## Position Sizing

- Risk 1% of account per trade
- Maximum 2 concurrent ORB positions

## Filters

- Only trade between 10:00–15:30 ET (no entries in first or last 30 min)
- Skip stocks with opening range width > 3% of price (too volatile)
- Skip stocks with opening range width < 0.3% of price (too tight)
- Minimum average daily volume: 1,000,000 shares
