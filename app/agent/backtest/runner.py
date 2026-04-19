"""
Backtest runner — thin orchestration layer.

Fetches historical data, generates entry signals, and runs vectorbt
Portfolio simulation with stop-loss / take-profit.

Supports both daily (1d) and intraday (5min) timeframes, and can run
discovered strategies from the pattern discovery pipeline.
"""

import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import vectorbt as vbt

from .strategies import STRATEGIES, SWING_STRATEGIES, DAY_STRATEGIES
from .results import serialize_results, save_run, BacktestConfig

logger = logging.getLogger(__name__)

# Ensure backend modules are importable
_backend_dir = os.path.join(os.path.dirname(__file__), "..", "..", "backend")
if _backend_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(_backend_dir))

# Map timeframe to vbt freq string
TIMEFRAME_FREQ = {
    "1d": "1D",
    "5min": "5T",
    "15min": "15T",
    "1h": "1h",
}


def _get_data_provider():
    """Lazy-import DataProvider to avoid circular imports at module level."""
    from data_provider import DataProvider
    from data_fetcher import MassiveDataFetcher
    from data_cache import DataCache

    return DataProvider(fetcher=MassiveDataFetcher(), cache=DataCache())


def _bars_to_dataframe(bars: List[Dict]) -> pd.DataFrame:
    """Convert list of bar dicts (from DataProvider) to a pandas DataFrame."""
    df = pd.DataFrame(bars)
    if "timestamp" in df.columns:
        df["date"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("date").sort_index()
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if "volume" in df.columns:
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
    return df


def _get_strategies_for_agent(agent_type: str = "") -> Dict:
    """Get strategy registry filtered by agent type."""
    if agent_type == "swing":
        return SWING_STRATEGIES
    elif agent_type == "day":
        return DAY_STRATEGIES
    return STRATEGIES


def _get_discovered_strategies() -> Dict:
    """Load discovered strategies from evolved rules if available."""
    discovered = {}
    try:
        from ..ml.discovery.evolver import StrategyEvolver

        evolver = StrategyEvolver()
        if evolver.load_evolved_rules():
            for rule in evolver.get_best_rules(n=5):
                key = f"discovered_rule_{rule.rule_id}"
                discovered[key] = evolver.rule_to_strategy(rule)
    except Exception as e:
        logger.debug(f"No discovered strategies available: {e}")
    return discovered


def get_available_strategies(agent_type: str = "") -> List[Dict]:
    """Return list of available strategies with metadata for the frontend.

    Args:
        agent_type: "swing", "day", or "" for all.

    Returns:
        List of {value, label, agent_type, source} dicts.
    """
    strategies = []

    for name in SWING_STRATEGIES:
        if agent_type in ("", "swing"):
            strategies.append({
                "value": name,
                "label": name.replace("_", " ").title(),
                "agent_type": "swing",
                "source": "builtin",
            })

    for name in DAY_STRATEGIES:
        if agent_type in ("", "day"):
            strategies.append({
                "value": name,
                "label": name.replace("_", " ").title(),
                "agent_type": "day",
                "source": "builtin",
            })

    # Add discovered rules
    for name in _get_discovered_strategies():
        strategies.append({
            "value": name,
            "label": f"Discovered: {name.replace('_', ' ').title()}",
            "agent_type": "day",
            "source": "discovered",
        })

    return strategies


def run_backtest(config: BacktestConfig) -> Dict:
    """
    Run a backtest for the given configuration.

    1. Fetch historical bars via DataProvider (supports timeframe)
    2. Generate entry signals using the strategy function
    3. Run vbt.Portfolio.from_signals with sl_stop / tp_stop
    4. Aggregate results across symbols
    """
    strategy_name = config["strategy"]
    timeframe = config.get("timeframe", "1d")
    agent_type = config.get("agent_type", "")

    # Build combined strategy registry (builtins + discovered)
    all_strategies = {**STRATEGIES, **_get_discovered_strategies()}

    if strategy_name not in all_strategies:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. "
            f"Available: {list(all_strategies.keys())}"
        )

    strategy_fn = all_strategies[strategy_name]
    symbols = config["symbols"]
    initial_capital = config.get("initial_capital", 100_000)
    commission = config.get("commission_pct", 0.001)
    slippage = config.get("slippage_pct", 0.0005)
    vbt_freq = TIMEFRAME_FREQ.get(timeframe, "1D")

    start_date = config.get("start_date", "")
    end_date = config.get("end_date", "")

    # Calculate days_back from date range
    if start_date and end_date:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        days_back = (end_dt - start_dt).days + 30  # Extra padding for indicator warmup
    else:
        days_back = 365

    data_provider = _get_data_provider()

    # Per-symbol capital allocation
    capital_per_symbol = initial_capital / len(symbols) if symbols else initial_capital

    all_portfolios = []
    symbol_results = {}

    for symbol in symbols:
        logger.info(f"Backtesting {strategy_name} on {symbol} ({timeframe})...")
        try:
            bars = data_provider.get_bars(symbol, timeframe, days_back=days_back)
            if not bars:
                logger.warning(f"No data for {symbol}, skipping")
                continue

            df = _bars_to_dataframe(bars)

            # Filter to date range
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]

            min_bars = 30 if timeframe == "1d" else 78  # At least 1 session for intraday
            if len(df) < min_bars:
                logger.warning(f"Insufficient data for {symbol} ({len(df)} bars), skipping")
                continue

            signals = strategy_fn(df)
            entries = signals["entries"]

            pf = vbt.Portfolio.from_signals(
                close=df["close"],
                open=df["open"],
                high=df["high"],
                low=df["low"],
                entries=entries,
                sl_stop=signals["sl_stop"],
                tp_stop=signals["tp_stop"],
                init_cash=capital_per_symbol,
                fees=commission,
                slippage=slippage,
                freq=vbt_freq,
            )

            all_portfolios.append(pf)
            symbol_results[symbol] = pf

        except Exception as e:
            logger.error(f"Error backtesting {symbol}: {e}", exc_info=True)
            continue

    if not all_portfolios:
        raise RuntimeError("No symbols produced valid backtest results")

    # Use the first portfolio if single symbol, otherwise combine
    if len(all_portfolios) == 1:
        combined_pf = all_portfolios[0]
    else:
        # For multi-symbol, we report per-symbol and pick the combined view
        combined_pf = all_portfolios[0]  # Primary for metrics
        # We'll aggregate in serialize_results

    result = serialize_results(combined_pf, config, symbol_results)
    save_run(result)
    return result


def run_discovered_backtest(config: BacktestConfig) -> List[Dict]:
    """Run backtests for all discovered (evolved) rules.

    Args:
        config: Base config with symbols, dates, capital. Strategy is ignored;
            all discovered rules are tested.

    Returns:
        List of backtest result dicts, one per discovered rule.
    """
    discovered = _get_discovered_strategies()
    if not discovered:
        raise RuntimeError("No discovered strategies available. Run discovery pipeline first.")

    results = []
    for name, fn in discovered.items():
        rule_config = dict(config)
        rule_config["strategy"] = name
        rule_config["timeframe"] = config.get("timeframe", "5min")
        rule_config["agent_type"] = "day"

        try:
            # Temporarily register the discovered strategy
            STRATEGIES[name] = fn
            result = run_backtest(rule_config)
            results.append(result)
        except Exception as e:
            logger.error(f"Discovered backtest failed for {name}: {e}")
        finally:
            STRATEGIES.pop(name, None)

    return results
