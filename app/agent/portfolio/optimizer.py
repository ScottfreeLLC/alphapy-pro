"""Portfolio optimizer using skfolio for cross-agent weight allocation.

Strategies:
- HRP (Hierarchical Risk Parity) — default, robust, no return estimation needed
- MeanRisk — mean-variance with CVaR risk measure
- BlackLitterman — uses agent confidence scores as analyst views

The optimizer fits on historical asset returns and produces per-asset weights
that the coordinator uses to size positions before execution.
"""

import logging
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Weight constraints
MIN_WEIGHT = 0.0
MAX_WEIGHT = 0.15  # Max 15% per asset
MIN_ASSETS_FOR_FIT = 3


class OptStrategy(str, Enum):
    HRP = "hrp"
    MEAN_RISK = "mean_risk"
    BLACK_LITTERMAN = "black_litterman"


class PortfolioOptimizer:
    """
    Portfolio-level weight optimizer using skfolio.

    Fits on historical daily returns for the symbols in both agents' watchlists,
    then adjusts signal position sizes based on optimized weights.
    """

    def __init__(
        self,
        strategy: OptStrategy = OptStrategy.HRP,
        max_weight: float = MAX_WEIGHT,
        min_weight: float = MIN_WEIGHT,
        rebalance_days: int = 20,
    ):
        self.strategy = strategy
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.rebalance_days = rebalance_days

        self._model: Optional[Any] = None
        self._weights: Optional[Dict[str, float]] = None
        self._fitted_at: Optional[datetime] = None
        self._returns_df: Optional[pd.DataFrame] = None
        self._fit_metrics: Optional[Dict] = None

    @property
    def is_fitted(self) -> bool:
        return self._weights is not None

    @property
    def weights(self) -> Dict[str, float]:
        return dict(self._weights) if self._weights else {}

    @property
    def needs_rebalance(self) -> bool:
        if self._fitted_at is None:
            return True
        days_since = (datetime.now() - self._fitted_at).days
        return days_since >= self.rebalance_days

    def fit(
        self,
        returns_df: pd.DataFrame,
        views: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """
        Fit the optimizer on historical asset returns.

        Args:
            returns_df: DataFrame of daily returns, columns=asset symbols.
                        Must have at least 60 rows and 3 columns.
            views: Optional dict of {symbol: expected_return} for BlackLitterman.
                   Typically derived from agent confidence scores.

        Returns:
            Dict with weights, strategy, fit metrics.
        """
        if len(returns_df.columns) < MIN_ASSETS_FOR_FIT:
            return {"error": f"Need at least {MIN_ASSETS_FOR_FIT} assets, got {len(returns_df.columns)}"}

        if len(returns_df) < 60:
            return {"error": f"Need at least 60 return observations, got {len(returns_df)}"}

        # Clean data
        returns_df = returns_df.replace([np.inf, -np.inf], np.nan).dropna(axis=1, thresh=int(len(returns_df) * 0.8))
        returns_df = returns_df.fillna(0.0)

        if len(returns_df.columns) < MIN_ASSETS_FOR_FIT:
            return {"error": "Too few assets after cleaning"}

        self._returns_df = returns_df

        try:
            if self.strategy == OptStrategy.HRP:
                result = self._fit_hrp(returns_df)
            elif self.strategy == OptStrategy.MEAN_RISK:
                result = self._fit_mean_risk(returns_df)
            elif self.strategy == OptStrategy.BLACK_LITTERMAN:
                result = self._fit_black_litterman(returns_df, views)
            else:
                return {"error": f"Unknown strategy: {self.strategy}"}

            self._fitted_at = datetime.now()
            self._fit_metrics = result
            logger.info(f"Portfolio optimizer fitted ({self.strategy.value}): {len(self._weights)} assets")
            return result

        except Exception as e:
            logger.error(f"Portfolio optimization failed: {e}", exc_info=True)
            return {"error": str(e)}

    def optimize_signals(
        self,
        signals: List[Dict],
        equity: float = 100000.0,
    ) -> List[Dict]:
        """
        Adjust signal position sizes using portfolio weights.

        Args:
            signals: List of signal dicts (must have 'symbol' and 'position_size_pct').
            equity: Total portfolio equity.

        Returns:
            Signals with adjusted position_size_pct.
        """
        if not self.is_fitted:
            return signals

        for signal in signals:
            symbol = signal.get("symbol", "")
            weight = self._weights.get(symbol, 0.0)

            if weight > 0:
                # Scale the original position size by the portfolio weight
                original_pct = signal.get("position_size_pct", 0.02)
                # Weight acts as a multiplier: higher weight = larger allocation
                # Normalize: if weight = max_weight, keep original size
                scale = weight / self.max_weight if self.max_weight > 0 else 1.0
                adjusted_pct = original_pct * min(2.0, max(0.25, scale))
                signal["position_size_pct"] = round(adjusted_pct, 4)
                signal["portfolio_weight"] = round(weight, 4)
            else:
                # Asset not in optimized universe — reduce to minimum
                signal["position_size_pct"] = round(signal.get("position_size_pct", 0.02) * 0.5, 4)
                signal["portfolio_weight"] = 0.0

        return signals

    def get_status(self) -> Dict:
        """Get optimizer status for API/frontend."""
        return {
            "strategy": self.strategy.value,
            "is_fitted": self.is_fitted,
            "fitted_at": self._fitted_at.isoformat() if self._fitted_at else None,
            "needs_rebalance": self.needs_rebalance,
            "rebalance_days": self.rebalance_days,
            "n_assets": len(self._weights) if self._weights else 0,
            "weights": self.weights,
            "max_weight": self.max_weight,
            "fit_metrics": self._fit_metrics,
        }

    def _fit_hrp(self, returns_df: pd.DataFrame) -> Dict:
        """Fit Hierarchical Risk Parity model."""
        from skfolio import RiskMeasure
        from skfolio.optimization import HierarchicalRiskParity

        model = HierarchicalRiskParity(
            risk_measure=RiskMeasure.CVAR,
            min_weights=self.min_weight,
            max_weights=self.max_weight,
            portfolio_params=dict(name="HRP-CVaR"),
        )
        model.fit(returns_df)
        self._model = model

        weights = dict(zip(returns_df.columns, model.weights_))
        self._weights = {k: round(v, 6) for k, v in weights.items() if v > 1e-6}

        # Evaluate on recent data
        portfolio = model.predict(returns_df.iloc[-60:])

        return {
            "strategy": "hrp",
            "n_assets": len(self._weights),
            "total_weight": round(sum(self._weights.values()), 4),
            "top_5": sorted(self._weights.items(), key=lambda x: -x[1])[:5],
            "annualized_return": _safe_attr(portfolio, "annualized_mean"),
            "annualized_vol": _safe_attr(portfolio, "annualized_standard_deviation"),
            "sharpe": _safe_attr(portfolio, "annualized_sharpe_ratio"),
        }

    def _fit_mean_risk(self, returns_df: pd.DataFrame) -> Dict:
        """Fit Mean-Risk (CVaR) model with Maximum Sharpe objective."""
        from skfolio import RiskMeasure, ObjectiveFunction
        from skfolio.optimization import MeanRisk

        model = MeanRisk(
            risk_measure=RiskMeasure.CVAR,
            objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
            min_weights=self.min_weight,
            max_weights=self.max_weight,
            portfolio_params=dict(name="MeanRisk-MaxSharpe"),
        )
        model.fit(returns_df)
        self._model = model

        weights = dict(zip(returns_df.columns, model.weights_))
        self._weights = {k: round(v, 6) for k, v in weights.items() if v > 1e-6}

        portfolio = model.predict(returns_df.iloc[-60:])

        return {
            "strategy": "mean_risk",
            "n_assets": len(self._weights),
            "total_weight": round(sum(self._weights.values()), 4),
            "top_5": sorted(self._weights.items(), key=lambda x: -x[1])[:5],
            "annualized_return": _safe_attr(portfolio, "annualized_mean"),
            "annualized_vol": _safe_attr(portfolio, "annualized_standard_deviation"),
            "sharpe": _safe_attr(portfolio, "annualized_sharpe_ratio"),
        }

    def _fit_black_litterman(
        self,
        returns_df: pd.DataFrame,
        views: Optional[Dict[str, float]] = None,
    ) -> Dict:
        """Fit Black-Litterman model with analyst views from agent confidence."""
        from skfolio import RiskMeasure, ObjectiveFunction
        from skfolio.optimization import MeanRisk
        from skfolio.prior import BlackLitterman

        # Build views: list of (asset, expected_return) tuples
        # If no explicit views, use equal-weight neutral views
        if views:
            # Convert views dict to the format skfolio expects
            # Views is a list of tuples: (asset_name, expected_annual_return)
            analyst_views = [
                [symbol, "==", expected_ret]
                for symbol, expected_ret in views.items()
                if symbol in returns_df.columns
            ]
        else:
            analyst_views = None

        prior_kwargs = {}
        if analyst_views:
            prior_kwargs["views"] = analyst_views

        try:
            model = MeanRisk(
                risk_measure=RiskMeasure.CVAR,
                objective_function=ObjectiveFunction.MAXIMIZE_RATIO,
                prior_estimator=BlackLitterman(**prior_kwargs) if prior_kwargs else None,
                min_weights=self.min_weight,
                max_weights=self.max_weight,
                portfolio_params=dict(name="BlackLitterman"),
            )
            model.fit(returns_df)
        except Exception as e:
            logger.warning(f"BlackLitterman failed ({e}), falling back to HRP")
            return self._fit_hrp(returns_df)

        self._model = model
        weights = dict(zip(returns_df.columns, model.weights_))
        self._weights = {k: round(v, 6) for k, v in weights.items() if v > 1e-6}

        portfolio = model.predict(returns_df.iloc[-60:])

        return {
            "strategy": "black_litterman",
            "n_assets": len(self._weights),
            "total_weight": round(sum(self._weights.values()), 4),
            "top_5": sorted(self._weights.items(), key=lambda x: -x[1])[:5],
            "views_count": len(analyst_views) if analyst_views else 0,
            "annualized_return": _safe_attr(portfolio, "annualized_mean"),
            "annualized_vol": _safe_attr(portfolio, "annualized_standard_deviation"),
            "sharpe": _safe_attr(portfolio, "annualized_sharpe_ratio"),
        }


def build_returns_df(
    data_provider: Any,
    symbols: List[str],
    days_back: int = 252,
) -> pd.DataFrame:
    """
    Build a returns DataFrame from the DataProvider for portfolio optimization.

    Args:
        data_provider: DataProvider instance with get_bars() method.
        symbols: List of ticker symbols.
        days_back: Number of calendar days of daily bar history.

    Returns:
        DataFrame of daily returns (columns = symbols).
    """
    prices = {}
    for symbol in symbols:
        try:
            bars = data_provider.get_bars(symbol, timeframe="1d", days_back=days_back)
            if bars and len(bars) >= 60:
                df = pd.DataFrame(bars)
                df["close"] = pd.to_numeric(df["close"], errors="coerce")
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    df = df.set_index("date")
                prices[symbol] = df["close"]
        except Exception as e:
            logger.warning(f"Failed to get bars for {symbol}: {e}")

    if len(prices) < MIN_ASSETS_FOR_FIT:
        logger.warning(f"Only {len(prices)} assets with sufficient data")
        return pd.DataFrame()

    prices_df = pd.DataFrame(prices)
    prices_df = prices_df.dropna(axis=1, thresh=int(len(prices_df) * 0.8))
    prices_df = prices_df.fillna(method="ffill").fillna(method="bfill")

    returns_df = prices_df.pct_change().iloc[1:]
    returns_df = returns_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    logger.info(f"Built returns matrix: {returns_df.shape[0]} days x {returns_df.shape[1]} assets")
    return returns_df


def _safe_attr(obj: Any, attr: str) -> Optional[float]:
    """Safely get a float attribute, returning None if missing or NaN."""
    try:
        val = getattr(obj, attr, None)
        if val is None:
            return None
        val = float(val)
        if np.isnan(val) or np.isinf(val):
            return None
        return round(val, 6)
    except Exception:
        return None
