"""Automated feature extraction using tsfresh and pycatch22.

Two modes:
- tsfresh (794 features): Offline training/discovery — thorough but slow.
- pycatch22 (22 features): Real-time inference — fast enough for 30s agent cycles.

Features are combined with existing hand-crafted intraday features from
app/agent/ml/intraday/features.py for maximum predictive power.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class AutoFeatureExtractor:
    """Automated time-series feature extraction for pattern discovery."""

    def __init__(self):
        self._tsfresh_relevant: Optional[List[str]] = None
        self._feature_importance: Optional[Dict[str, float]] = None

    def extract_tsfresh(
        self,
        bars_df: pd.DataFrame,
        target: Optional[pd.Series] = None,
        window_size: int = 20,
    ) -> pd.DataFrame:
        """Generate features per window using tsfresh with relevance filtering.

        Args:
            bars_df: OHLCV DataFrame.
            target: Optional forward return series for relevance testing.
            window_size: Rolling window size for feature extraction.

        Returns:
            DataFrame of relevant features (one row per bar window).
        """
        from tsfresh import extract_features
        from tsfresh.utilities.dataframe_functions import roll_time_series

        close = bars_df["close"].values.astype(float)
        volume = bars_df["volume"].values.astype(float)
        high = bars_df["high"].values.astype(float)
        low = bars_df["low"].values.astype(float)

        n = len(bars_df)
        if n < window_size + 1:
            logger.warning(f"Insufficient data ({n} bars) for tsfresh extraction")
            return pd.DataFrame()

        # Build tsfresh input format: id (window), time, value columns
        rows = []
        for i in range(window_size, n):
            window_id = i
            for j in range(window_size):
                bar_idx = i - window_size + j
                rows.append({
                    "id": window_id,
                    "time": j,
                    "close": close[bar_idx],
                    "volume": volume[bar_idx],
                    "range": high[bar_idx] - low[bar_idx],
                    "return": (close[bar_idx] - close[max(0, bar_idx - 1)]) / close[max(0, bar_idx - 1)]
                    if close[max(0, bar_idx - 1)] > 0 else 0.0,
                })

        ts_df = pd.DataFrame(rows)

        try:
            features = extract_features(
                ts_df,
                column_id="id",
                column_sort="time",
                n_jobs=1,
                disable_progressbar=True,
                default_fc_parameters={
                    "mean": None,
                    "variance": None,
                    "skewness": None,
                    "kurtosis": None,
                    "abs_energy": None,
                    "mean_abs_change": None,
                    "longest_strike_above_mean": None,
                    "longest_strike_below_mean": None,
                    "count_above_mean": None,
                    "count_below_mean": None,
                    "last_location_of_maximum": None,
                    "first_location_of_maximum": None,
                    "last_location_of_minimum": None,
                    "first_location_of_minimum": None,
                    "percentage_of_reoccurring_datapoints_to_all_datapoints": None,
                    "ratio_beyond_r_sigma": [{"r": 1}, {"r": 2}],
                    "linear_trend": [{"attr": "slope"}, {"attr": "intercept"}, {"attr": "rvalue"}],
                    "autocorrelation": [{"lag": 1}, {"lag": 5}, {"lag": 10}],
                    "quantile": [{"q": 0.25}, {"q": 0.75}],
                    "number_peaks": [{"n": 3}, {"n": 5}],
                },
            )

            # Drop columns with all NaN/inf
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.dropna(axis=1, how="all")
            features = features.fillna(0.0)

            # Filter relevant features if target is provided
            if target is not None and len(target) == len(features):
                features = self._filter_relevant(features, target)

            self._tsfresh_relevant = list(features.columns)
            logger.info(f"tsfresh extracted {features.shape[1]} features for {features.shape[0]} windows")
            return features

        except Exception as e:
            logger.error(f"tsfresh extraction failed: {e}")
            return pd.DataFrame()

    def extract_catch22(self, bars_df: pd.DataFrame, window_size: int = 20) -> pd.DataFrame:
        """Fast 22-feature extraction for real-time use.

        ~100x faster than tsfresh. Suitable for 30s agent cycles.

        Args:
            bars_df: OHLCV DataFrame.
            window_size: Rolling window size.

        Returns:
            DataFrame with 22 features per window.
        """
        import pycatch22

        close = bars_df["close"].values.astype(float)
        n = len(close)

        if n < window_size + 1:
            return pd.DataFrame()

        all_features = []

        for i in range(window_size, n):
            window = close[i - window_size:i]
            try:
                result = pycatch22.catch22_all(window)
                feature_dict = dict(zip(result["names"], result["values"]))
                feature_dict["bar_index"] = i
                all_features.append(feature_dict)
            except Exception:
                continue

        if not all_features:
            return pd.DataFrame()

        features_df = pd.DataFrame(all_features).set_index("bar_index")
        features_df = features_df.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        logger.info(f"catch22 extracted {features_df.shape[1]} features for {features_df.shape[0]} windows")
        return features_df

    def combine_with_existing(
        self,
        auto_features: pd.DataFrame,
        existing_features: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge auto-discovered features with hand-crafted intraday features.

        Aligns on index and concatenates columns, handling mismatched lengths.
        """
        if auto_features.empty:
            return existing_features
        if existing_features.empty:
            return auto_features

        # Align on common indices
        common = auto_features.index.intersection(existing_features.index)
        if len(common) == 0:
            # Fall back to positional alignment if indices don't match
            min_len = min(len(auto_features), len(existing_features))
            auto_trimmed = auto_features.iloc[-min_len:].reset_index(drop=True)
            exist_trimmed = existing_features.iloc[-min_len:].reset_index(drop=True)
            combined = pd.concat([exist_trimmed, auto_trimmed], axis=1)
        else:
            combined = pd.concat(
                [existing_features.loc[common], auto_features.loc[common]], axis=1
            )

        # Remove duplicate columns
        combined = combined.loc[:, ~combined.columns.duplicated()]
        combined = combined.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        logger.info(f"Combined features: {combined.shape[1]} columns, {combined.shape[0]} rows")
        return combined

    def get_feature_importance(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> Dict[str, float]:
        """Rank features by predictive power using mutual information + permutation importance.

        Args:
            features: Feature matrix.
            target: Target variable (e.g. forward returns or binary labels).

        Returns:
            Dict mapping feature name -> importance score.
        """
        from sklearn.feature_selection import mutual_info_regression

        if len(features) != len(target) or features.empty:
            return {}

        # Align
        common = features.index.intersection(target.index)
        if len(common) < 10:
            return {}

        X = features.loc[common].values
        y = target.loc[common].values

        try:
            mi = mutual_info_regression(X, y, random_state=42, n_neighbors=5)
            importance = dict(zip(features.columns, mi))
            self._feature_importance = dict(
                sorted(importance.items(), key=lambda x: x[1], reverse=True)
            )
            return self._feature_importance
        except Exception as e:
            logger.error(f"Feature importance calculation failed: {e}")
            return {}

    def _filter_relevant(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        fdr_level: float = 0.05,
    ) -> pd.DataFrame:
        """Filter features using statistical hypothesis testing against target."""
        from tsfresh import select_features

        try:
            # Align target to features index
            aligned_target = target.loc[features.index]
            relevant = select_features(features, aligned_target, fdr_level=fdr_level)
            if relevant.empty:
                # Fall back to top features by variance if none pass the test
                variances = features.var().sort_values(ascending=False)
                top_cols = variances.head(min(50, len(variances))).index.tolist()
                return features[top_cols]
            return relevant
        except Exception as e:
            logger.warning(f"Feature selection failed, returning all: {e}")
            return features
