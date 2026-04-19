---
name: gap_fill
timeframes: [5min]
risk_per_trade: 0.01
max_positions: 2
enabled: true
tags: [intraday, gap, mean-reversion]
---

# Gap Fill Strategy

## Entry Conditions (ALL must be met)

1. **Moderate gap**: Stock has gapped 1–4% from previous close at the open.

2. **Fade direction confirmed**: After the first 15 minutes, price is moving back toward the previous close (filling the gap). For gap-up: price is declining. For gap-down: price is rising.

3. **VWAP slope**: VWAP is sloping in the gap-fill direction (down for gap-up fill, up for gap-down fill), confirming institutional flow supports the fill.

4. **Volume pattern**: Declining volume on the initial gap move, increasing volume on the fill move.

5. **No strong catalyst**: The gap is not driven by earnings, M&A, or FDA news (those gaps are less likely to fill).

## Exit Conditions

- **Stop loss**: Place stop at the intraday high (gap-up fade) or intraday low (gap-down fade) plus 0.2% buffer.
- **Take profit**: Primary target is 80% gap fill (not full fill — leave margin for error). Secondary target is full gap fill to previous close.
- **Partial exit**: Close 60% at the 80% fill level, let 40% ride to full fill with breakeven stop.
- **Time stop**: Close any remaining position by 14:00 ET.

## Position Sizing

- Risk 1% of account per trade
- Maximum 2 concurrent gap fill positions

## Filters

- Gap size between 1% and 4% (too small = not worth it, too large = may not fill)
- Minimum average daily volume: 1,000,000 shares
- Skip if gap is in the direction of the daily trend (trend gaps fill less often)
- Skip earnings gaps and known catalyst gaps
