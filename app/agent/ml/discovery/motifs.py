"""STUMPY-based motif discovery for unsupervised pattern detection on 5-min bars.

Uses matrix profiles to find recurring price subsequences without predefined
pattern definitions. Discovered motifs are clustered, evaluated for profitability,
and stored in a persistent library for real-time matching.
"""

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "models")


@dataclass
class Motif:
    """A discovered recurring price subsequence."""
    motif_id: int
    window_size: int
    template: np.ndarray  # z-normalized price subsequence
    occurrences: int = 0
    avg_forward_return: float = 0.0
    win_rate: float = 0.0
    sharpe: float = 0.0
    last_seen: Optional[str] = None
    cluster_id: int = -1
    symbols: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "motif_id": self.motif_id,
            "window_size": self.window_size,
            "template": self.template.tolist(),
            "occurrences": self.occurrences,
            "avg_forward_return": round(self.avg_forward_return, 6),
            "win_rate": round(self.win_rate, 4),
            "sharpe": round(self.sharpe, 4),
            "last_seen": self.last_seen,
            "cluster_id": self.cluster_id,
            "symbols": self.symbols,
        }


class MotifDiscoverer:
    """Discovers recurring price patterns using STUMPY matrix profiles."""

    def __init__(self, top_k: int = 10, min_occurrences: int = 5):
        self.top_k = top_k
        self.min_occurrences = min_occurrences
        self.motif_library: List[Motif] = []
        self._next_id = 0

    def discover(
        self,
        bars_df: pd.DataFrame,
        window_sizes: List[int] = None,
        symbol: str = "",
    ) -> List[Motif]:
        """Run STUMPY matrix profile across multiple window sizes.

        Args:
            bars_df: OHLCV DataFrame with 'close' column.
            window_sizes: Subsequence lengths to search (in bars).
                Default [20, 40, 60] = 100min, 200min, 300min at 5-min bars.
            symbol: Symbol name for tracking.

        Returns:
            List of discovered Motif objects.
        """
        import stumpy

        if window_sizes is None:
            window_sizes = [20, 40, 60]

        close = bars_df["close"].values.astype(float)
        if len(close) < max(window_sizes) * 3:
            logger.warning(f"Insufficient data ({len(close)} bars) for motif discovery")
            return []

        discovered = []

        for m in window_sizes:
            if len(close) < m * 2:
                continue

            try:
                mp = stumpy.stump(close, m)
                profile = mp[:, 0].astype(float)
                indices = mp[:, 1].astype(int)

                # Find top-K motifs (lowest matrix profile values = most similar pairs)
                sorted_idx = np.argsort(profile)

                seen = set()
                motif_count = 0

                for idx in sorted_idx:
                    if motif_count >= self.top_k:
                        break

                    nn_idx = indices[idx]
                    # Skip trivially overlapping pairs
                    if abs(idx - nn_idx) < m:
                        continue

                    pair_key = (min(idx, nn_idx), max(idx, nn_idx))
                    if pair_key in seen:
                        continue
                    seen.add(pair_key)

                    # Extract and z-normalize the motif template
                    subsequence = close[idx:idx + m]
                    template = self._z_normalize(subsequence)

                    motif = Motif(
                        motif_id=self._next_id,
                        window_size=m,
                        template=template,
                        occurrences=2,
                        last_seen=datetime.now().isoformat(),
                        symbols=[symbol] if symbol else [],
                    )
                    self._next_id += 1
                    discovered.append(motif)
                    motif_count += 1

            except Exception as e:
                logger.error(f"STUMPY error for window_size={m}: {e}")

        logger.info(f"Discovered {len(discovered)} motifs across {len(window_sizes)} window sizes")
        return discovered

    def find_cross_symbol_motifs(
        self,
        symbols_data: Dict[str, pd.DataFrame],
        window_size: int = 40,
    ) -> List[Motif]:
        """Find patterns that repeat across different stocks using STUMPY AB-joins.

        Args:
            symbols_data: Dict mapping symbol -> OHLCV DataFrame.
            window_size: Subsequence length.

        Returns:
            Cross-symbol motifs.
        """
        import stumpy

        symbols = list(symbols_data.keys())
        if len(symbols) < 2:
            return []

        cross_motifs = []

        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                sym_a, sym_b = symbols[i], symbols[j]
                ts_a = symbols_data[sym_a]["close"].values.astype(float)
                ts_b = symbols_data[sym_b]["close"].values.astype(float)

                if len(ts_a) < window_size * 2 or len(ts_b) < window_size * 2:
                    continue

                try:
                    ab = stumpy.stump(ts_a, window_size, ts_b)
                    profile = ab[:, 0].astype(float)

                    # Top match
                    best_idx = np.argmin(profile)
                    nn_idx = int(ab[best_idx, 1])

                    template = self._z_normalize(ts_a[best_idx:best_idx + window_size])
                    motif = Motif(
                        motif_id=self._next_id,
                        window_size=window_size,
                        template=template,
                        occurrences=2,
                        last_seen=datetime.now().isoformat(),
                        symbols=[sym_a, sym_b],
                    )
                    self._next_id += 1
                    cross_motifs.append(motif)

                except Exception as e:
                    logger.warning(f"AB-join error {sym_a}/{sym_b}: {e}")

        logger.info(f"Found {len(cross_motifs)} cross-symbol motifs")
        return cross_motifs

    def cluster_motifs(
        self,
        motifs: List[Motif],
        eps: float = 1.0,
        min_samples: int = 2,
    ) -> List[Motif]:
        """Group similar motifs using DBSCAN on z-normalized subsequences.

        Each cluster becomes a "discovered pattern" candidate.
        """
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler

        if len(motifs) < min_samples:
            return motifs

        # Group by window_size since we can only compare same-length subsequences
        by_window = {}
        for m in motifs:
            by_window.setdefault(m.window_size, []).append(m)

        for window_size, group in by_window.items():
            if len(group) < min_samples:
                continue

            templates = np.array([m.template for m in group])
            scaler = StandardScaler()
            scaled = scaler.fit_transform(templates)

            clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
            labels = clustering.fit_predict(scaled)

            for motif, label in zip(group, labels):
                motif.cluster_id = int(label)

        return motifs

    def evaluate_motifs(
        self,
        motifs: List[Motif],
        bars_df: pd.DataFrame,
        forward_bars: int = 10,
    ) -> List[Motif]:
        """Evaluate motif profitability by measuring forward returns after occurrences.

        Args:
            motifs: Motifs to evaluate.
            bars_df: Full OHLCV DataFrame.
            forward_bars: Number of bars to measure forward return.

        Returns:
            Motifs with updated stats, filtered to profitable ones.
        """
        import stumpy

        close = bars_df["close"].values.astype(float)
        evaluated = []

        for motif in motifs:
            m = motif.window_size
            if len(close) < m + forward_bars:
                continue

            try:
                # Find all occurrences using STUMPY match
                distances = stumpy.mass(motif.template, close)
                threshold = np.percentile(distances, 5)  # Top 5% matches
                match_indices = np.where(distances <= threshold)[0]

                # Filter out overlapping matches
                match_indices = self._remove_overlapping(match_indices, m)

                if len(match_indices) < self.min_occurrences:
                    continue

                # Measure forward returns
                returns = []
                for idx in match_indices:
                    end_idx = idx + m + forward_bars
                    if end_idx >= len(close):
                        continue
                    entry = close[idx + m]
                    exit_ = close[end_idx]
                    ret = (exit_ - entry) / entry
                    returns.append(ret)

                if not returns:
                    continue

                returns_arr = np.array(returns)
                motif.occurrences = len(returns_arr)
                motif.avg_forward_return = float(np.mean(returns_arr))
                motif.win_rate = float(np.mean(returns_arr > 0))
                motif.sharpe = float(
                    np.mean(returns_arr) / np.std(returns_arr)
                    if np.std(returns_arr) > 0 else 0.0
                )

                evaluated.append(motif)

            except Exception as e:
                logger.warning(f"Motif evaluation error (id={motif.motif_id}): {e}")

        # Filter to profitable motifs
        profitable = [m for m in evaluated if m.avg_forward_return > 0 and m.win_rate > 0.5]
        logger.info(
            f"Evaluated {len(evaluated)} motifs, {len(profitable)} profitable "
            f"(avg return > 0, win rate > 50%)"
        )
        return profitable

    def match_current_bars(
        self,
        bars_df: pd.DataFrame,
    ) -> List[Dict]:
        """Check if current bars match any motif in the library.

        Args:
            bars_df: Recent OHLCV bars.

        Returns:
            List of matches with motif_id, distance, and motif stats.
        """
        import stumpy

        if not self.motif_library:
            return []

        close = bars_df["close"].values.astype(float)
        matches = []

        for motif in self.motif_library:
            if len(close) < motif.window_size:
                continue

            try:
                # Check the most recent window
                recent = close[-motif.window_size:]
                recent_norm = self._z_normalize(recent)
                distance = np.linalg.norm(recent_norm - motif.template)

                # Threshold: distance < 2.0 is a reasonable match for z-normalized series
                if distance < 2.0:
                    matches.append({
                        "motif_id": motif.motif_id,
                        "distance": round(float(distance), 4),
                        "window_size": motif.window_size,
                        "avg_forward_return": motif.avg_forward_return,
                        "win_rate": motif.win_rate,
                        "sharpe": motif.sharpe,
                        "occurrences": motif.occurrences,
                    })
            except Exception:
                continue

        return sorted(matches, key=lambda x: x["distance"])

    def save_motif_library(self, path: Optional[str] = None) -> str:
        """Persist discovered motifs to disk."""
        import joblib

        os.makedirs(MODELS_DIR, exist_ok=True)
        if path is None:
            date_str = datetime.now().strftime("%Y%m%d")
            path = os.path.join(MODELS_DIR, f"motif_library_{date_str}.joblib")

        data = {
            "motifs": self.motif_library,
            "next_id": self._next_id,
            "saved_at": datetime.now().isoformat(),
        }
        joblib.dump(data, path)
        logger.info(f"Saved motif library ({len(self.motif_library)} motifs) to {path}")
        return path

    def load_motif_library(self, path: Optional[str] = None) -> bool:
        """Load motif library from disk."""
        import joblib

        if path is None:
            # Find most recent library file
            if not os.path.exists(MODELS_DIR):
                return False
            files = sorted(
                [f for f in os.listdir(MODELS_DIR) if f.startswith("motif_library_")],
                reverse=True,
            )
            if not files:
                return False
            path = os.path.join(MODELS_DIR, files[0])

        if not os.path.exists(path):
            return False

        data = joblib.load(path)
        self.motif_library = data["motifs"]
        self._next_id = data.get("next_id", len(self.motif_library))
        logger.info(f"Loaded motif library ({len(self.motif_library)} motifs) from {path}")
        return True

    def get_library_summary(self) -> Dict:
        """Get summary of the current motif library."""
        if not self.motif_library:
            return {"total_motifs": 0, "motifs": []}

        return {
            "total_motifs": len(self.motif_library),
            "motifs": [m.to_dict() for m in self.motif_library],
            "window_sizes": list(set(m.window_size for m in self.motif_library)),
            "avg_win_rate": round(
                np.mean([m.win_rate for m in self.motif_library]), 4
            ),
            "avg_sharpe": round(
                np.mean([m.sharpe for m in self.motif_library]), 4
            ),
        }

    @staticmethod
    def _z_normalize(ts: np.ndarray) -> np.ndarray:
        """Z-normalize a time series."""
        std = np.std(ts)
        if std == 0:
            return np.zeros_like(ts)
        return (ts - np.mean(ts)) / std

    @staticmethod
    def _remove_overlapping(indices: np.ndarray, window: int) -> np.ndarray:
        """Remove overlapping match indices, keeping the closest match."""
        if len(indices) == 0:
            return indices
        sorted_idx = np.sort(indices)
        kept = [sorted_idx[0]]
        for idx in sorted_idx[1:]:
            if idx - kept[-1] >= window:
                kept.append(idx)
        return np.array(kept)
