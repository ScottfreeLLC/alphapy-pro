---
name: pivot_pattern_entry
timeframes: [1d]
risk_per_trade: 0.02
max_positions: 4
enabled: true
tags: [pivots, patterns, harmonic]
---

# Pivot Pattern Entry Strategy

## Entry Conditions (ALL must be met)

1. **Bullish pattern detected**: The pattern analysis has detected one or more bullish patterns (Bullish Gartley, Bullish ABCD, Bullish Three-Drive, Bullish Wolfe Wave, Bullish Expansion, or Bullish Squeeze).

2. **Overall sentiment bullish**: The pattern analyzer's overall_sentiment for this stock is "bullish" (more bullish patterns than bearish).

3. **Pattern confluence**: At least 2 different bullish patterns are detected simultaneously, providing stronger confirmation.

4. **Price confirmation**: Current price is within the pattern's target zone and showing follow-through (close above open on the most recent bar).

## Short Entry (Mirror)

For bearish patterns, the mirror conditions apply:
- Bearish pattern detected (Bearish Gartley, Bearish ABCD, etc.)
- Overall sentiment is bearish
- At least 2 bearish patterns detected
- Current price shows bearish follow-through

## Exit Conditions

- **Stop loss**: For longs, place stop at the pattern's invalidation level (typically the most recent pivot low). For shorts, above the most recent pivot high.
- **Take profit**: Target the pattern's projected completion point. For harmonic patterns, this is typically a Fibonacci extension.
- **Partial exit**: Take 50% off at 1:1 risk/reward, let remainder run to full target.

## Position Sizing

- Risk 2% of account per trade
- Maximum 4 concurrent positions (patterns are higher conviction)

## Filters

- Require at least 100 bars of data for reliable pivot detection
- Skip if the pattern is more than 5 bars old (stale signal)
- Higher conviction when multiple pattern types align
