"""
Serialize vectorbt results to JSON-friendly dicts and persist to SQLite.
"""

import json
import logging
import os
import sqlite3
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "backtests", "runs.db")


class BacktestConfig(TypedDict, total=False):
    strategy: str
    symbols: List[str]
    start_date: str
    end_date: str
    initial_capital: float
    commission_pct: float
    slippage_pct: float
    timeframe: str       # "1d" or "5min"
    agent_type: str      # "swing" or "day"


def _safe_float(val: Any) -> Optional[float]:
    """Convert numpy/pandas scalar to Python float, returning None on failure."""
    if val is None:
        return None
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return None
        return round(f, 4)
    except (TypeError, ValueError):
        return None


def serialize_results(pf, config: Dict, symbol_results: Dict = None) -> Dict:
    """Extract vectorbt Portfolio results into a JSON-serializable dict."""
    run_id = str(uuid.uuid4())[:8]

    # Core metrics from the portfolio
    total_return = _safe_float(pf.total_return() * 100)
    sharpe = _safe_float(pf.sharpe_ratio())
    max_dd = _safe_float(pf.max_drawdown() * 100)
    final_value = _safe_float(pf.final_value())

    # Trade-level metrics
    trades = pf.trades
    total_trades = int(trades.count()) if trades.count() > 0 else 0
    win_rate = _safe_float(trades.win_rate() * 100) if total_trades > 0 else None
    profit_factor = _safe_float(trades.profit_factor()) if total_trades > 0 else None

    # Trade PnL stats
    if total_trades > 0:
        pnl_series = trades.pnl.values
        avg_pnl = _safe_float(np.mean(pnl_series))
        best_trade = _safe_float(np.max(pnl_series))
        worst_trade = _safe_float(np.min(pnl_series))
    else:
        avg_pnl = best_trade = worst_trade = None

    # Equity curve
    equity = pf.value()
    equity_curve = []
    for dt, val in equity.items():
        equity_curve.append({
            "date": str(dt.date()) if hasattr(dt, "date") else str(dt),
            "equity": round(float(val), 2),
        })

    # Trade records
    trade_records = []
    if total_trades > 0:
        try:
            readable = trades.records_readable
            for _, row in readable.iterrows():
                record = {}
                for col in readable.columns:
                    v = row[col]
                    if isinstance(v, (pd.Timestamp, datetime)):
                        record[col] = str(v)
                    elif isinstance(v, (np.integer,)):
                        record[col] = int(v)
                    elif isinstance(v, (np.floating, float)):
                        record[col] = round(float(v), 4) if not np.isnan(v) else None
                    else:
                        record[col] = str(v) if v is not None else None
                trade_records.append(record)
        except Exception as e:
            logger.warning(f"Could not serialize trade records: {e}")

    # Per-symbol summary
    symbol_summaries = {}
    if symbol_results:
        for sym, sym_pf in symbol_results.items():
            sym_trades = sym_pf.trades
            symbol_summaries[sym] = {
                "total_return_pct": _safe_float(sym_pf.total_return() * 100),
                "total_trades": int(sym_trades.count()) if sym_trades.count() > 0 else 0,
                "win_rate": _safe_float(sym_trades.win_rate() * 100) if sym_trades.count() > 0 else None,
                "final_equity": _safe_float(sym_pf.final_value()),
            }

    return {
        "run_id": run_id,
        "config": dict(config),
        "created_at": datetime.now().isoformat(),
        "metrics": {
            "total_return_pct": total_return,
            "total_trades": total_trades,
            "win_rate": win_rate,
            "sharpe_ratio": sharpe,
            "max_drawdown_pct": max_dd,
            "profit_factor": profit_factor,
            "avg_trade_pnl": avg_pnl,
            "best_trade": best_trade,
            "worst_trade": worst_trade,
            "final_equity": final_value,
        },
        "equity_curve": equity_curve,
        "trades": trade_records,
        "symbol_summaries": symbol_summaries,
    }


# ---------------------------------------------------------------------------
# SQLite persistence
# ---------------------------------------------------------------------------

def _ensure_db():
    """Create the runs table if it doesn't exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            config TEXT NOT NULL,
            results TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_run(result: Dict):
    """Persist a backtest run to SQLite."""
    _ensure_db()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT OR REPLACE INTO runs (run_id, config, results, created_at) VALUES (?, ?, ?, ?)",
        (
            result["run_id"],
            json.dumps(result["config"]),
            json.dumps(result),
            result["created_at"],
        ),
    )
    conn.commit()
    conn.close()
    logger.info(f"Saved backtest run {result['run_id']}")


def list_runs() -> List[Dict]:
    """Return summary of all past runs (most recent first)."""
    _ensure_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute(
        "SELECT run_id, config, created_at FROM runs ORDER BY created_at DESC LIMIT 50"
    )
    runs = []
    for row in cursor:
        cfg = json.loads(row[1])
        runs.append({
            "run_id": row[0],
            "strategy": cfg.get("strategy", ""),
            "symbols": cfg.get("symbols", []),
            "start_date": cfg.get("start_date", ""),
            "end_date": cfg.get("end_date", ""),
            "created_at": row[2],
        })
    conn.close()
    return runs


def get_run(run_id: str) -> Optional[Dict]:
    """Retrieve full results for a specific run."""
    _ensure_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.execute("SELECT results FROM runs WHERE run_id = ?", (run_id,))
    row = cursor.fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return None
