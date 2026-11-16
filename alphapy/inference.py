################################################################################
#
# Package   : AlphaPy
# Module    : inference
# Created   : November 15, 2026
#
# Copyright 2026 ScottFree Analytics LLC
# Mark Conway & Robert D. Scott II
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################

"""Score new rows with a trained run, exactly as training scored its test rows.

``load_run(run_dir)`` reads what training saved in ``<run_dir>/model``: the manifest, every fitted algorithm, the
feature map (feature selection supports) and the fitted feature pipeline. ``predict_proba(frame)`` turns raw rows
(the columns training loaded, before transforms and drops) into features with the pipeline as fitted on the training
rows, gives each algorithm its own columns through the same selection code training used, and returns each
algorithm's probability of the positive class and the blend's (the plain mean of the members', as in training).
Nothing is written.

Window algorithms (the ``sequence`` section) read each row's earlier rows in its group. The frame supplies them, or
``history`` does: earlier raw rows that are not scored. ``TrainedRun.training_tail`` holds each group's last training
rows, the history training gave its test rows; pass it only when the frame's rows directly follow training's.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import polars as pl

from alphapy.estimators import is_window_algo
from alphapy.globals import ModelType
from alphapy.model import Model
from alphapy.model import algo_matrix
from alphapy.sequence import window_block
from alphapy.sequence import window_columns

logger = logging.getLogger(__name__)

BLEND = "blend"


class TrainedRun:
    """A trained run loaded for scoring: see the module documentation."""

    def __init__(self, run_dir, manifest, pipeline, feature_map, estimators):
        self.run_dir = Path(run_dir)
        self.manifest = manifest
        self.pipeline = pipeline
        self.estimators = estimators
        self.algorithms = list(manifest["algorithms"])
        self.blend = list(manifest.get("blend") or [])
        self.classification = manifest["model_type"] == str(ModelType.classification)
        self.sequence = manifest.get("sequence")
        # the model state training's column selection reads (alphapy.model.algo_matrix)
        gate = Model({"algorithms": self.algorithms, "n_tabular": manifest.get("n_tabular")})
        gate.feature_map = feature_map
        gate.feature_names = list(manifest["feature_names"])
        gate.window_names = list(manifest.get("window_names") or [])
        gate.fnames_algo = {a: list(n) for a, n in manifest.get("feature_names_by_algorithm", {}).items()}
        gate.support = {a: np.asarray(s, dtype=bool) for a, s in manifest.get("rfe_support", {}).items()}
        self._gate = gate

    @property
    def training_tail(self):
        """Each group's last training rows (window columns only), or None without the sequence section."""
        return self.pipeline.history

    def features(self, frame):
        """The tabular features of ``frame``'s rows, named as in training (before any per-algorithm selection)."""
        X = self.pipeline.transform(self.pipeline.prepare(self.pipeline.conform(frame)))
        return pd.DataFrame(X, columns=self.pipeline.feature_names, index=_index(frame))

    def predict_proba(self, frame, algorithms=None, history=None):
        """Each algorithm's probability of the positive class for ``frame``'s rows, and the blend's.

        Parameters
        ----------
        frame : pandas.DataFrame or polars.DataFrame
            Raw rows with the columns training loaded; other columns are ignored.
        algorithms : list of str, optional
            The algorithms to return (``"blend"`` for the blend); all of them and the blend when omitted.
        history : pandas.DataFrame or polars.DataFrame, optional
            Earlier raw rows for window algorithms' windows; they are not scored.

        Returns
        -------
        pandas.DataFrame
            One column per algorithm (and ``blend``), indexed like ``frame``.
        """
        if not self.classification:
            raise ValueError("This run is not a classification; use predict for its values")
        return self.score(frame, algorithms, history)[1]

    def predict(self, frame, algorithms=None, history=None):
        """Each algorithm's predicted label (classification) or value, and the blend's, indexed like ``frame``."""
        return self.score(frame, algorithms, history)[0]

    def score(self, frame, algorithms=None, history=None):
        """(predictions, probabilities) as two DataFrames; probabilities is None for a regression."""
        names = self._names(algorithms)
        scored = list(dict.fromkeys([a for a in names if a != BLEND] + (self.blend if BLEND in names else [])))
        raw = self.pipeline.conform(frame)
        X = self.pipeline.transform(self.pipeline.prepare(raw))
        if self.sequence:       # training stacked the tabular columns and the windows as one float32 matrix
            X = np.hstack([np.asarray(X, dtype=np.float32), self._windows(raw, history)])
        preds, probas = {}, {}
        for algo in sorted(scored, key=is_window_algo):     # tabular first, as in training (torch after LightGBM)
            X_algo, _ = algo_matrix(self._gate, algo, X)
            estimator = self.estimators[algo]
            pred = estimator.predict(X_algo)
            if pred.ndim > 1 and pred.shape[1] > 1:
                pred = np.argmax(pred, axis=1)
            preds[algo] = pred
            if self.classification:
                probas[algo] = estimator.predict_proba(X_algo)[:, 1]
        if BLEND in names:      # the plain mean of the members, in training's order
            members = sorted(self.blend, key=is_window_algo)
            if self.classification:
                probas[BLEND] = np.mean([probas[a] for a in members], axis=0)
                preds[BLEND] = np.round(probas[BLEND], 0).astype(int)
            else:
                preds[BLEND] = np.round(np.mean([preds[a] for a in members], axis=0), 0).astype(int)
        index = _index(frame)
        predictions = pd.DataFrame({a: preds[a] for a in names}, index=index)
        probabilities = pd.DataFrame({a: probas[a] for a in names}, index=index) if self.classification else None
        return predictions, probabilities

    def _names(self, algorithms):
        available = self.algorithms + ([BLEND] if self.blend else [])
        if algorithms is None:
            return available
        unknown = [a for a in algorithms if a not in available]
        if unknown:
            raise ValueError(f"The run has no algorithm {unknown}; it has {available}")
        return [a for a in available if a in algorithms]

    def _windows(self, raw, history):
        spec = self.sequence
        columns = window_columns(spec)
        frame = raw.select(columns).to_pandas()
        past = None
        if history is not None:
            if not isinstance(history, pl.DataFrame):
                history = pl.from_pandas(pd.DataFrame(history).reset_index(drop=True))
            past = history.select(columns).cast({c: self.pipeline.raw_schema[c] for c in columns}).to_pandas()
            keys = [spec["group"], spec["order"]]   # a row in both is the frame's
            seen = pd.MultiIndex.from_frame(frame[keys])
            past = past[~pd.MultiIndex.from_frame(past[keys]).isin(seen)]
            frame = pd.concat([past, frame], ignore_index=True)
        block = window_block(frame, spec)
        return block if past is None else block[len(past):]


def _index(frame):
    return frame.index if isinstance(frame, pd.DataFrame) else pd.RangeIndex(len(frame))


def load_run(run_dir):
    """Load a trained run for scoring (see the module documentation)."""
    model_dir = Path(run_dir) / "model"
    pipeline_path = model_dir / "feature_pipeline.pkl"
    if not pipeline_path.exists():
        raise FileNotFoundError(f"{run_dir} has no saved feature pipeline ({pipeline_path.name}). Runs trained before "
                                "the feature steps were fitted on training rows only and saved cannot be scored "
                                "again as training scored them; train the run again.")
    manifest = json.loads((model_dir / "manifest.json").read_text())
    pipeline = joblib.load(pipeline_path)
    feature_map = joblib.load(model_dir / "feature_map.pkl")
    estimators = {a: joblib.load(model_dir / "estimators" / f"{a}.pkl") for a in manifest["algorithms"]}
    logger.info("Loaded run %s: %s", run_dir, ", ".join(manifest["algorithms"]))
    return TrainedRun(run_dir, manifest, pipeline, feature_map, estimators)
