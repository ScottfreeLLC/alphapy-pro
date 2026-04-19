"""Portfolio module for both backtest and live trading.

Provides:
- LivePortfolio / LivePosition: real-time portfolio management with Alpaca integration
- Legacy event-driven backtest API (Trade, gen_portfolios) preserved for
  backwards compatibility. New code should use ``alphapy.backtest`` which
  drives vectorbt directly.
"""

from .live import LivePortfolio, LivePosition
from .legacy import Trade, gen_portfolios

__all__ = [
    "LivePortfolio",
    "LivePosition",
    "Trade",
    "gen_portfolios",
]
