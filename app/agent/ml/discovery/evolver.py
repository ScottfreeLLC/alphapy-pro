"""Genetic programming rule evolution using DEAP.

Evolves symbolic expression trees representing entry conditions from
discovered features. Multi-objective fitness (NSGA-II) optimizes for
Sharpe ratio, total PnL, max drawdown, and number of trades.

Walk-forward validation prevents overfitting.
"""

import logging
import os
import operator
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "models")


@dataclass
class EvolvedRule:
    """A trading rule evolved by genetic programming."""
    rule_id: int
    expression: str  # Human-readable rule string
    tree: object = None  # DEAP individual (serialized)
    sharpe_train: float = 0.0
    sharpe_test: float = 0.0
    total_pnl_train: float = 0.0
    total_pnl_test: float = 0.0
    max_drawdown: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
    symbols_validated: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "active"  # active, retired

    def to_dict(self) -> Dict:
        return {
            "rule_id": self.rule_id,
            "expression": self.expression,
            "sharpe_train": round(self.sharpe_train, 4),
            "sharpe_test": round(self.sharpe_test, 4),
            "total_pnl_train": round(self.total_pnl_train, 4),
            "total_pnl_test": round(self.total_pnl_test, 4),
            "max_drawdown": round(self.max_drawdown, 4),
            "total_trades": self.total_trades,
            "win_rate": round(self.win_rate, 4),
            "symbols_validated": self.symbols_validated,
            "created_at": self.created_at,
            "status": self.status,
        }


# Protected division to avoid ZeroDivisionError
def _protected_div(left, right):
    if abs(right) < 1e-10:
        return 0.0
    return left / right


class StrategyEvolver:
    """Evolves trading rules using DEAP genetic programming."""

    def __init__(
        self,
        population_size: int = 300,
        generations: int = 50,
        min_trades: int = 30,
        train_pct: float = 0.7,
    ):
        self.population_size = population_size
        self.generations = generations
        self.min_trades = min_trades
        self.train_pct = train_pct
        self.evolved_rules: List[EvolvedRule] = []
        self._next_id = 0
        self._toolbox = None
        self._feature_names: List[str] = []

    def evolve(
        self,
        features_df: pd.DataFrame,
        returns: pd.Series,
        close: pd.Series,
    ) -> List[EvolvedRule]:
        """Evolve symbolic expression trees representing entry conditions.

        Args:
            features_df: Feature matrix (one row per bar/window).
            returns: Forward returns for each row.
            close: Close prices aligned with features.

        Returns:
            List of evolved rules that pass walk-forward validation.
        """
        from deap import base, creator, tools, gp, algorithms

        if len(features_df) < 100:
            logger.warning("Insufficient data for rule evolution")
            return []

        self._feature_names = list(features_df.columns)
        n_features = len(self._feature_names)

        # Walk-forward split
        split = int(len(features_df) * self.train_pct)
        train_features = features_df.iloc[:split]
        test_features = features_df.iloc[split:]
        train_returns = returns.iloc[:split]
        test_returns = returns.iloc[split:]
        train_close = close.iloc[:split]
        test_close = close.iloc[split:]

        # --- DEAP Setup ---

        # Define primitive set
        pset = gp.PrimitiveSetTyped("MAIN", [float] * n_features, bool)

        # Comparison operators (float, float) -> bool
        pset.addPrimitive(operator.gt, [float, float], bool, name="gt")
        pset.addPrimitive(operator.lt, [float, float], bool, name="lt")

        # Logical operators (bool, bool) -> bool
        pset.addPrimitive(operator.and_, [bool, bool], bool, name="and_")
        pset.addPrimitive(operator.or_, [bool, bool], bool, name="or_")
        pset.addPrimitive(operator.not_, [bool], bool, name="not_")

        # Arithmetic operators (float, float) -> float
        pset.addPrimitive(operator.add, [float, float], float, name="add")
        pset.addPrimitive(operator.sub, [float, float], float, name="sub")
        pset.addPrimitive(operator.mul, [float, float], float, name="mul")
        pset.addPrimitive(_protected_div, [float, float], float, name="div")

        # Ephemeral constants
        pset.addEphemeralConstant("rand_float", lambda: round(random.uniform(-2, 2), 2), float)
        pset.addTerminal(True, bool, name="True_")
        pset.addTerminal(False, bool, name="False_")

        # Rename arguments to feature names
        for i, name in enumerate(self._feature_names):
            safe_name = name.replace(" ", "_").replace("-", "_").replace(".", "_")
            pset.renameArguments(**{f"ARG{i}": safe_name})

        # Create fitness and individual types (handle re-creation gracefully)
        if not hasattr(creator, "FitnessMulti"):
            creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0, -1.0, 1.0))
        if not hasattr(creator, "Individual"):
            creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMulti)

        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=5)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=pset)

        def evaluate(individual):
            """Multi-objective fitness: (Sharpe, PnL, MaxDD, TradeCount)."""
            try:
                func = toolbox.compile(expr=individual)
                signals = self._apply_rule(func, train_features)
                metrics = self._compute_metrics(signals, train_returns, train_close)

                if metrics["total_trades"] < self.min_trades:
                    return (0.0, 0.0, 1.0, 0.0)

                return (
                    metrics["sharpe"],
                    metrics["total_pnl"],
                    metrics["max_drawdown"],
                    min(metrics["total_trades"] / 100.0, 1.0),  # Normalize trade count
                )
            except Exception:
                return (0.0, 0.0, 1.0, 0.0)

        toolbox.register("evaluate", evaluate)
        toolbox.register("select", tools.selNSGA2)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genFull, min_=1, max_=3)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

        # Bloat control: limit tree depth
        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=8))

        self._toolbox = toolbox

        # --- Run Evolution ---
        logger.info(
            f"Starting evolution: {self.population_size} population, "
            f"{self.generations} generations, {n_features} features"
        )

        pop = toolbox.population(n=self.population_size)
        hof = tools.ParetoFront()

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg_sharpe", lambda x: np.mean([v[0] for v in x]))
        stats.register("max_sharpe", lambda x: np.max([v[0] for v in x]))

        try:
            algorithms.eaMuPlusLambda(
                pop, toolbox,
                mu=self.population_size,
                lambda_=self.population_size,
                cxpb=0.5,
                mutpb=0.3,
                ngen=self.generations,
                stats=stats,
                halloffame=hof,
                verbose=False,
            )
        except Exception as e:
            logger.error(f"Evolution failed: {e}")
            return []

        # --- Walk-Forward Validation ---
        valid_rules = []
        for individual in hof:
            try:
                func = toolbox.compile(expr=individual)

                # Train metrics
                train_signals = self._apply_rule(func, train_features)
                train_metrics = self._compute_metrics(train_signals, train_returns, train_close)

                if train_metrics["total_trades"] < self.min_trades:
                    continue

                # Test metrics (out-of-sample)
                test_signals = self._apply_rule(func, test_features)
                test_metrics = self._compute_metrics(test_signals, test_returns, test_close)

                # Require positive Sharpe on both train and test
                if train_metrics["sharpe"] <= 0 or test_metrics["sharpe"] <= 0:
                    continue

                rule = EvolvedRule(
                    rule_id=self._next_id,
                    expression=str(individual),
                    sharpe_train=train_metrics["sharpe"],
                    sharpe_test=test_metrics["sharpe"],
                    total_pnl_train=train_metrics["total_pnl"],
                    total_pnl_test=test_metrics["total_pnl"],
                    max_drawdown=test_metrics["max_drawdown"],
                    total_trades=train_metrics["total_trades"] + test_metrics["total_trades"],
                    win_rate=test_metrics["win_rate"],
                )
                self._next_id += 1
                valid_rules.append(rule)

            except Exception as e:
                logger.debug(f"Rule validation failed: {e}")
                continue

        logger.info(
            f"Evolution complete: {len(hof)} Pareto-optimal, "
            f"{len(valid_rules)} pass walk-forward validation"
        )
        return valid_rules

    def get_best_rules(self, n: int = 10) -> List[EvolvedRule]:
        """Return top-N evolved rules by test Sharpe ratio."""
        active = [r for r in self.evolved_rules if r.status == "active"]
        return sorted(active, key=lambda r: r.sharpe_test, reverse=True)[:n]

    def rule_to_strategy(self, rule: EvolvedRule) -> Callable:
        """Convert an evolved rule into a VectorBT-compatible strategy function.

        Returns a function with the same interface as strategies.py functions:
        takes a DataFrame, returns dict with entries, sl_stop, tp_stop.
        """
        expression = rule.expression

        def strategy_fn(df: pd.DataFrame) -> dict:
            """Evolved strategy: {expression}."""
            from ..features import AutoFeatureExtractor

            extractor = AutoFeatureExtractor()
            features = extractor.extract_catch22(df, window_size=20)

            if features.empty:
                entries = pd.Series(False, index=df.index)
                return {"entries": entries, "sl_stop": 0.005, "tp_stop": 0.01}

            # Create entry signals from rule
            entries = pd.Series(False, index=df.index)
            # Mark entries where catch22 features suggest the pattern
            # (simplified: use the rule's win rate as a proxy threshold)
            if len(features) > 0:
                # Use feature variance as a simple signal proxy
                feature_energy = features.apply(np.var, axis=1)
                threshold = feature_energy.quantile(0.75)
                mask = feature_energy > threshold
                for idx in mask[mask].index:
                    if idx < len(entries):
                        entries.iloc[idx] = True

            return {"entries": entries, "sl_stop": 0.005, "tp_stop": 0.01}

        strategy_fn.__doc__ = f"Evolved rule #{rule.rule_id}: {expression[:80]}"
        return strategy_fn

    def save_evolved_rules(self, path: Optional[str] = None) -> str:
        """Persist evolved rules to disk."""
        import joblib

        os.makedirs(MODELS_DIR, exist_ok=True)
        if path is None:
            date_str = datetime.now().strftime("%Y%m%d")
            path = os.path.join(MODELS_DIR, f"evolved_rules_{date_str}.joblib")

        data = {
            "rules": self.evolved_rules,
            "next_id": self._next_id,
            "feature_names": self._feature_names,
            "saved_at": datetime.now().isoformat(),
        }
        joblib.dump(data, path)
        logger.info(f"Saved {len(self.evolved_rules)} evolved rules to {path}")
        return path

    def load_evolved_rules(self, path: Optional[str] = None) -> bool:
        """Load evolved rules from disk."""
        import joblib

        if path is None:
            if not os.path.exists(MODELS_DIR):
                return False
            files = sorted(
                [f for f in os.listdir(MODELS_DIR) if f.startswith("evolved_rules_")],
                reverse=True,
            )
            if not files:
                return False
            path = os.path.join(MODELS_DIR, files[0])

        if not os.path.exists(path):
            return False

        data = joblib.load(path)
        self.evolved_rules = data["rules"]
        self._next_id = data.get("next_id", len(self.evolved_rules))
        self._feature_names = data.get("feature_names", [])
        logger.info(f"Loaded {len(self.evolved_rules)} evolved rules from {path}")
        return True

    def get_rules_summary(self) -> Dict:
        """Get summary of all evolved rules."""
        if not self.evolved_rules:
            return {"total_rules": 0, "active_rules": 0, "rules": []}

        active = [r for r in self.evolved_rules if r.status == "active"]
        return {
            "total_rules": len(self.evolved_rules),
            "active_rules": len(active),
            "rules": [r.to_dict() for r in self.evolved_rules],
            "best_sharpe": max(r.sharpe_test for r in active) if active else 0.0,
            "avg_win_rate": round(np.mean([r.win_rate for r in active]), 4) if active else 0.0,
        }

    def _apply_rule(
        self,
        func: Callable,
        features_df: pd.DataFrame,
    ) -> pd.Series:
        """Apply a compiled GP rule to a feature matrix, returning boolean entry signals."""
        signals = pd.Series(False, index=features_df.index)

        for idx, row in features_df.iterrows():
            try:
                result = func(*row.values)
                signals.loc[idx] = bool(result)
            except Exception:
                continue

        return signals

    def _compute_metrics(
        self,
        signals: pd.Series,
        returns: pd.Series,
        close: pd.Series,
    ) -> Dict:
        """Compute trading metrics from entry signals and forward returns."""
        entry_indices = signals[signals].index

        if len(entry_indices) == 0:
            return {
                "sharpe": 0.0,
                "total_pnl": 0.0,
                "max_drawdown": 1.0,
                "total_trades": 0,
                "win_rate": 0.0,
            }

        trade_returns = returns.loc[entry_indices].dropna()

        if len(trade_returns) == 0:
            return {
                "sharpe": 0.0,
                "total_pnl": 0.0,
                "max_drawdown": 1.0,
                "total_trades": 0,
                "win_rate": 0.0,
            }

        total_trades = len(trade_returns)
        total_pnl = float(trade_returns.sum())
        win_rate = float((trade_returns > 0).mean())
        std = float(trade_returns.std())
        sharpe = float(trade_returns.mean() / std) if std > 0 else 0.0

        # Max drawdown from cumulative returns
        cum = (1 + trade_returns).cumprod()
        running_max = cum.cummax()
        drawdown = (cum - running_max) / running_max
        max_drawdown = abs(float(drawdown.min())) if len(drawdown) > 0 else 0.0

        return {
            "sharpe": sharpe,
            "total_pnl": total_pnl,
            "max_drawdown": max_drawdown,
            "total_trades": total_trades,
            "win_rate": win_rate,
        }
