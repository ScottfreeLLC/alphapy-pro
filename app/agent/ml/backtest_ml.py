"""ML backtest harness: walk-forward evaluation of meta-labeling model."""

import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .features import build_feature_matrix
from .labeling import get_daily_volatility, get_events, triple_barrier_labels
from .meta_model import MetaModel

logger = logging.getLogger(__name__)


def run_ml_backtest(
    config: Dict,
    data_fetcher=None,
    stock_data: Optional[Dict] = None,
) -> Dict:
    """
    Run ML backtest with walk-forward validation.

    Config keys:
        strategy: str — strategy name (must be in STRATEGIES)
        symbols: list[str]
        start_date: str
        end_date: str
        initial_capital: float
        train_pct: float (default 0.7)
        pt_sl: tuple (default (1.0, 1.0))
        vertical_barrier_periods: int (default 10)
        ml_threshold: float (default 0.5)

    Returns:
        Dict with backtest results comparing strategy-alone vs strategy+ML
    """
    # Import strategies
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "backtest"))
    from ..backtest.strategies import STRATEGIES

    strategy_name = config["strategy"]
    if strategy_name not in STRATEGIES:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(STRATEGIES.keys())}")

    strategy_fn = STRATEGIES[strategy_name]
    symbols = config["symbols"]
    initial_capital = config.get("initial_capital", 100000)
    train_pct = config.get("train_pct", 0.7)
    pt_sl = tuple(config.get("pt_sl", [1.0, 1.0]))
    vertical_periods = config.get("vertical_barrier_periods", 10)
    ml_threshold = config.get("ml_threshold", 0.5)

    all_results = []

    for symbol in symbols:
        try:
            result = _backtest_symbol(
                symbol=symbol,
                strategy_fn=strategy_fn,
                stock_data=stock_data,
                data_fetcher=data_fetcher,
                start_date=config.get("start_date"),
                end_date=config.get("end_date"),
                initial_capital=initial_capital / len(symbols),
                train_pct=train_pct,
                pt_sl=pt_sl,
                vertical_periods=vertical_periods,
                ml_threshold=ml_threshold,
            )
            if result:
                all_results.append(result)
        except Exception as e:
            logger.error(f"ML backtest error for {symbol}: {e}", exc_info=True)

    if not all_results:
        raise RuntimeError("No backtest results generated")

    # Aggregate results
    return _aggregate_results(all_results, config)


def _backtest_symbol(
    symbol: str,
    strategy_fn,
    stock_data: Optional[Dict],
    data_fetcher,
    start_date: Optional[str],
    end_date: Optional[str],
    initial_capital: float,
    train_pct: float,
    pt_sl: tuple,
    vertical_periods: int,
    ml_threshold: float,
) -> Optional[Dict]:
    """Run ML backtest for a single symbol."""
    # Get data
    df = None
    if stock_data and symbol in stock_data:
        df = stock_data[symbol].copy()
    elif data_fetcher:
        from ..backtest.runner import _fetch_data
        df = _fetch_data(data_fetcher, symbol, start_date, end_date)

    if df is None or len(df) < 100:
        logger.warning(f"Insufficient data for {symbol}")
        return None

    # Ensure date index
    if "date" in df.columns:
        df.index = pd.to_datetime(df["date"])

    # Generate strategy signals
    signals = strategy_fn(df)
    entries = signals["entries"]
    sl_stop = signals["sl_stop"]
    tp_stop = signals["tp_stop"]

    entry_count = entries.sum()
    if entry_count < 10:
        logger.warning(f"{symbol}: only {entry_count} signals, need 10+")
        return None

    # Build features
    feature_matrix = build_feature_matrix(df)

    # Build triple barrier labels
    close = df["close"].astype(float)
    volatility = get_daily_volatility(close)
    events = get_events(close, entries, vertical_periods)
    labels = triple_barrier_labels(close, events, pt_sl, volatility=volatility)

    if len(labels) < 20:
        logger.warning(f"{symbol}: only {len(labels)} labeled events")
        return None

    # Walk-forward split
    split_idx = int(len(labels) * train_pct)
    train_labels = labels.iloc[:split_idx]
    test_labels = labels.iloc[split_idx:]

    train_features = feature_matrix.reindex(train_labels.index).dropna()
    test_features = feature_matrix.reindex(test_labels.index).dropna()

    # Align
    common_train = train_features.index.intersection(train_labels.index)
    common_test = test_features.index.intersection(test_labels.index)

    if len(common_train) < 20 or len(common_test) < 5:
        logger.warning(f"{symbol}: insufficient aligned data for train/test split")
        return None

    train_X = train_features.loc[common_train]
    train_y = train_labels.loc[common_train, "label"]
    test_X = test_features.loc[common_test]
    test_y = test_labels.loc[common_test, "label"]
    test_returns = test_labels.loc[common_test, "ret"]

    # Train meta-model
    model = MetaModel()
    train_metrics = model.train(train_X, train_y)

    if "error" in train_metrics:
        logger.warning(f"{symbol}: training failed — {train_metrics['error']}")
        return None

    # Evaluate on test set
    test_probas = []
    for idx in test_X.index:
        feature_dict = test_X.loc[idx].to_dict()
        proba, _ = model.predict(feature_dict)
        test_probas.append(proba)

    test_probas = pd.Series(test_probas, index=test_X.index)

    # Strategy-only returns (all signals)
    strategy_returns = test_returns.copy()

    # ML-filtered returns (only signals where proba >= threshold)
    ml_mask = test_probas >= ml_threshold
    ml_returns = test_returns[ml_mask]

    # Calculate metrics for both
    strategy_metrics = _compute_metrics(strategy_returns, initial_capital)
    ml_metrics = _compute_metrics(ml_returns, initial_capital)

    return {
        "symbol": symbol,
        "strategy_only": strategy_metrics,
        "strategy_plus_ml": ml_metrics,
        "train_metrics": train_metrics,
        "total_signals": len(test_returns),
        "ml_filtered_signals": int(ml_mask.sum()),
        "filter_rate": round(1 - ml_mask.mean(), 4) if len(ml_mask) > 0 else 0,
        "ml_threshold": ml_threshold,
    }


def _compute_metrics(returns: pd.Series, initial_capital: float) -> Dict:
    """Compute performance metrics from a return series."""
    if len(returns) == 0:
        return {
            "total_trades": 0, "win_rate": 0, "total_return_pct": 0,
            "sharpe": 0, "max_drawdown_pct": 0, "profit_factor": 0,
        }

    wins = returns[returns > 0]
    losses = returns[returns <= 0]

    win_rate = len(wins) / len(returns) if len(returns) > 0 else 0
    total_return = returns.sum()

    # Sharpe
    mean_ret = returns.mean()
    std_ret = returns.std()
    sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0

    # Profit factor
    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Max drawdown
    cumulative = returns.cumsum()
    peak = cumulative.cummax()
    drawdown = peak - cumulative
    max_dd = drawdown.max() * 100

    return {
        "total_trades": len(returns),
        "win_rate": round(float(win_rate), 4),
        "total_return_pct": round(float(total_return * 100), 2),
        "sharpe": round(float(sharpe), 2),
        "max_drawdown_pct": round(float(max_dd), 2),
        "profit_factor": round(float(profit_factor), 2) if profit_factor != float("inf") else "inf",
        "avg_return_pct": round(float(returns.mean() * 100), 4),
    }


def _aggregate_results(results: List[Dict], config: Dict) -> Dict:
    """Aggregate per-symbol results into a summary."""
    strategy_trades = sum(r["strategy_only"]["total_trades"] for r in results)
    ml_trades = sum(r["strategy_plus_ml"]["total_trades"] for r in results)

    strategy_returns = [r["strategy_only"]["total_return_pct"] for r in results]
    ml_returns = [r["strategy_plus_ml"]["total_return_pct"] for r in results]

    return {
        "run_id": f"ml_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "config": config,
        "symbols": [r["symbol"] for r in results],
        "per_symbol": results,
        "summary": {
            "strategy_only": {
                "total_trades": strategy_trades,
                "avg_return_pct": round(np.mean(strategy_returns), 2) if strategy_returns else 0,
                "total_return_pct": round(sum(strategy_returns), 2),
            },
            "strategy_plus_ml": {
                "total_trades": ml_trades,
                "avg_return_pct": round(np.mean(ml_returns), 2) if ml_returns else 0,
                "total_return_pct": round(sum(ml_returns), 2),
            },
            "improvement": {
                "trades_filtered_pct": round((1 - ml_trades / max(1, strategy_trades)) * 100, 1),
                "return_diff_pct": round(sum(ml_returns) - sum(strategy_returns), 2),
            },
        },
        "timestamp": datetime.now().isoformat(),
    }
