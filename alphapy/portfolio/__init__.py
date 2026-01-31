"""Portfolio module for both backtest and live trading.

This module provides:
- LivePortfolio: Real-time portfolio management with Alpaca integration
- Position tracking synchronized with live broker
- P&L calculations using Polars DataFrames
- Backtest portfolio functions from portfolio.py
"""

from .live import LivePortfolio, LivePosition

# Re-export backtest portfolio functions from the original portfolio.py
# Import using relative import from parent package
import importlib.util
import os

_portfolio_py = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'portfolio.py')
_spec = importlib.util.spec_from_file_location("_portfolio_backtest", _portfolio_py)
_portfolio_backtest = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_portfolio_backtest)

# Export the backtest functions
gen_portfolios = _portfolio_backtest.gen_portfolios
Trade = _portfolio_backtest.Trade

__all__ = [
    "LivePortfolio",
    "LivePosition",
    "gen_portfolios",
    "Trade",
]
