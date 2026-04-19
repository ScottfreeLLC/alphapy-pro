"""Walk-forward training pipeline for the intraday pattern classifier.

Workflow:
1. Fetch historical 5-min bars from DataProvider/Massive
2. Label bars with heuristic pattern labeler
3. Build feature matrix
4. 70/30 walk-forward split (chronological, no shuffle)
5. Train XGBoost classifier
6. Evaluate on held-out test set
7. Save model to disk
"""

import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .classifier import IntradayClassifier
from .features import build_multi_session_features
from .patterns import IntradayPattern, _detect_session_breaks, label_multi_session

logger = logging.getLogger(__name__)


class IntradayTrainer:
    """
    Orchestrates training of the intraday pattern classifier.

    Can be called directly or via the API endpoint.
    """

    def __init__(
        self,
        data_fetcher: Optional[Any] = None,
        data_provider: Optional[Any] = None,
    ):
        self.data_fetcher = data_fetcher
        self.data_provider = data_provider

    def train(
        self,
        symbols: List[str],
        days_back: int = 90,
        train_pct: float = 0.7,
        save_model: bool = True,
    ) -> Dict:
        """
        Run the full training pipeline.

        Args:
            symbols: List of ticker symbols to train on.
            days_back: Number of calendar days of history to fetch.
            train_pct: Fraction of data for training (rest is evaluation).
            save_model: Whether to persist the trained model.

        Returns:
            Dict with training results, metrics, and model path.
        """
        logger.info(f"Starting intraday training: {len(symbols)} symbols, {days_back} days")

        # 1. Fetch 5-min bars for all symbols
        all_bars = self._fetch_bars(symbols, days_back)
        if not all_bars:
            return {"error": "No 5-min bar data available", "symbols_tried": symbols}

        # 2. Build combined feature matrix + labels
        all_features = []
        all_labels = []
        symbol_stats = {}

        for symbol, df in all_bars.items():
            if len(df) < 100:
                logger.warning(f"{symbol}: only {len(df)} bars, skipping")
                continue

            session_breaks = _detect_session_breaks(df)

            # Label
            labels = label_multi_session(df, session_breaks)

            # Features
            features = build_multi_session_features(df, session_breaks)

            if len(features) == 0:
                continue

            # Align (features may have fewer rows)
            common_idx = features.index.intersection(labels.index)
            features = features.loc[common_idx]
            labels = labels.loc[common_idx]

            all_features.append(features)
            all_labels.append(labels)

            dist = labels.value_counts().to_dict()
            symbol_stats[symbol] = {
                "bars": len(df),
                "labeled_bars": len(labels),
                "pattern_distribution": {IntradayPattern(k).name: v for k, v in dist.items()},
            }
            logger.info(f"{symbol}: {len(df)} bars, {len(labels)} labeled")

        if not all_features:
            return {"error": "No features could be built", "symbol_stats": symbol_stats}

        X = pd.concat(all_features)
        y = pd.concat(all_labels)

        logger.info(f"Combined dataset: {X.shape[0]} bars x {X.shape[1]} features")

        # 3. Walk-forward split (chronological)
        split_idx = int(len(X) * train_pct)
        train_X, eval_X = X.iloc[:split_idx], X.iloc[split_idx:]
        train_y, eval_y = y.iloc[:split_idx], y.iloc[split_idx:]

        logger.info(f"Split: {len(train_X)} train / {len(eval_X)} eval")

        # 4. Train classifier
        classifier = IntradayClassifier()
        metrics = classifier.train(train_X, train_y, eval_X, eval_y)

        if "error" in metrics:
            return metrics

        # 5. Save model
        model_path = ""
        if save_model:
            model_path = classifier.save()

        return {
            "status": "success",
            "model_path": model_path,
            "total_bars": len(X),
            "train_bars": len(train_X),
            "eval_bars": len(eval_X),
            "symbols": list(all_bars.keys()),
            "symbol_stats": symbol_stats,
            "metrics": metrics,
        }

    def _fetch_bars(
        self,
        symbols: List[str],
        days_back: int,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch 5-min bars for each symbol."""
        result = {}
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        for symbol in symbols:
            df = self._fetch_symbol_bars(symbol, start_date, end_date)
            if df is not None and len(df) > 0:
                result[symbol] = df

        logger.info(f"Fetched 5-min bars for {len(result)}/{len(symbols)} symbols")
        return result

    def _fetch_symbol_bars(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """Fetch 5-min bars for a single symbol."""
        # Try DataProvider first (has caching)
        if self.data_provider:
            try:
                bars = self.data_provider.get_bars(
                    symbol, timeframe="5min", days_back=90, use_cache=True
                )
                if bars:
                    df = pd.DataFrame(bars)
                    return self._normalize_df(df)
            except Exception as e:
                logger.warning(f"DataProvider fetch failed for {symbol}: {e}")

        # Fall back to direct fetcher
        if self.data_fetcher:
            try:
                df = self.data_fetcher.fetch_bars(symbol, "5min", start_date, end_date)
                if df is not None and not df.empty:
                    return self._normalize_df(df)
            except Exception as e:
                logger.warning(f"DataFetcher fetch failed for {symbol}: {e}")

        logger.warning(f"No 5-min data available for {symbol}")
        return None

    @staticmethod
    def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names and types."""
        # Ensure required columns exist
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            # Try common renames
            renames = {"o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
            df = df.rename(columns={k: v for k, v in renames.items() if k in df.columns})

        if not required.issubset(df.columns):
            return pd.DataFrame()

        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)

        # Ensure date column exists
        if "date" not in df.columns and "t" in df.columns:
            df["date"] = pd.to_datetime(df["t"], unit="ms")
        elif "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])

        df = df.dropna(subset=["close"])
        df = df.reset_index(drop=True)
        return df
