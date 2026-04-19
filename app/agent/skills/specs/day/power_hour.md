---
name: power_hour
timeframes: [5min]
risk_per_trade: 0.01
max_positions: 2
enabled: true
tags: [intraday, momentum, closing]
---

# Power Hour Momentum Strategy

## Entry Conditions (ALL must be met)

1. **Time window**: Entry only between 15:00–15:45 ET (power hour).

2. **Intraday trend**: Stock has a clear intraday direction — up at least 1% from open (for longs) or down at least 1% (for shorts).

3. **Volume acceleration**: Volume in the 14:30–15:00 period is at least 1.5x the midday average (11:00–14:00), signaling institutional positioning into the close.

4. **Price near session extreme**: For longs, price is within 0.5% of the session high. For shorts, price is within 0.5% of the session low.

5. **VWAP confirmation**: For longs, price is above VWAP. For shorts, price is below VWAP.

## Exit Conditions

- **Stop loss**: 0.5% adverse move from entry (tight stop for end-of-day trades).
- **Take profit**: Hold into the close (15:59 ET) — power hour moves tend to accelerate into the bell.
- **Hard close**: All positions closed by 15:58 ET — absolutely no overnight holds from this strategy.

## Position Sizing

- Risk 1% of account per trade
- Maximum 2 concurrent power hour positions

## Filters

- Only trade stocks that have been trending intraday (not rangebound)
- Minimum intraday range: 1.5% (high - low) / open
- Minimum average daily volume: 1,000,000 shares
- Skip if major economic announcement is scheduled after market close
