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

``load_run(run_dir)`` reads what training saved in ``<run_dir>/model``: the manifest, the feature map (feature
selection supports) and the fitted feature pipeline; each fitted algorithm is loaded when it is first asked for.
``predict_proba(frame)`` turns raw rows (the columns training loaded, before transforms and drops) into features with
the pipeline as fitted on the training rows, gives each algorithm its own columns through the same selection code
training used, and returns each algorithm's probability of the positive class and the blend's (the plain mean of the
members', as in training). Nothing is written.

Window algorithms (the ``sequence`` section) read each row's ``window - 1`` earlier rows in its group. The frame
supplies them, or ``history`` does: earlier raw rows that are not scored. A row with fewer earlier rows than that is
refused, because its window would be padded where training's had values; ``allow_short_history=True`` scores it
anyway. ``TrainedRun.training_tail`` holds each group's last training rows, the history training gave its test rows,
and ``history_from_training(frame)`` keeps the groups whose rows in ``frame`` come after them.
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
from alphapy.sequence import order_key
from alphapy.sequence import window_block
from alphapy.sequence import window_columns

logger = logging.getLogger(__name__)

BLEND = "blend"


class TrainedRun:
    """A trained run loaded for scoring: see the module documentation."""

    def __init__(self, run_dir, manifest, pipeline, feature_map):
        self.run_dir = Path(run_dir)
        self.manifest = manifest
        self.pipeline = pipeline
        self.algorithms = list(manifest["algorithms"])
        self.blend = list(manifest.get("blend") or [])
        self.classification = manifest["model_type"] == str(ModelType.classification)
        self.sequence = manifest.get("sequence")
        self._estimators = {}
        # the model state that training's per-algorithm column selection (alphapy.model.algo_matrix) reads
        selection = Model({"algorithms": self.algorithms, "n_tabular": manifest.get("n_tabular")})
        selection.feature_map = feature_map
        selection.feature_names = list(manifest["feature_names"])
        selection.window_names = list(manifest.get("window_names") or [])
        selection.fnames_algo = {a: list(n) for a, n in manifest.get("feature_names_by_algorithm", {}).items()}
        selection.support = {a: np.asarray(s, dtype=bool) for a, s in manifest.get("rfe_support", {}).items()}
        self._selection = selection

    def estimator(self, algo):
        """The fitted estimator of ``algo``, loaded from the run on first use."""
        if algo not in self._estimators:
            self._estimators[algo] = joblib.load(self.run_dir / "model" / "estimators" / f"{algo}.pkl")
        return self._estimators[algo]

    @property
    def training_tail(self):
        """Each group's last training rows (window columns only), or None without the sequence section."""
        return self.pipeline.history

    def history_from_training(self, frame):
        """The training tail of each group whose rows in ``frame`` all come after training's last row in that group,
        by the order column: the history training gave its test rows. None without the sequence section."""
        tail = self.training_tail
        if tail is None:
            return None
        group, order = self.sequence["group"], self.sequence["order"]
        rows = self.pipeline.conform(frame).select([group, order]).to_pandas()
        first = rows.assign(_key=order_key(rows[order]).to_numpy()).groupby(group)["_key"].min()
        last = tail.assign(_key=order_key(tail[order]).to_numpy()).groupby(group)["_key"].max()
        follows = [g for g in first.index if g in last.index and first[g] > last[g]]
        return tail[tail[group].isin(follows)].reset_index(drop=True)

    def features(self, frame):
        """The tabular features of ``frame``'s rows, named as in training (before any per-algorithm selection)."""
        X = self.pipeline.transform(self.pipeline.prepare(self.pipeline.conform(frame)))
        return pd.DataFrame(X, columns=self.pipeline.feature_names, index=_index(frame))

    def predict_proba(self, frame, algorithms=None, history=None, allow_short_history=False):
        """Each algorithm's probability of the positive class for ``frame``'s rows, and the blend's.

        Parameters
        ----------
        frame : pandas.DataFrame or polars.DataFrame
            Raw rows with the columns training loaded; other columns are ignored.
        algorithms : list of str, optional
            The algorithms to return (``"blend"`` for the blend); all of them and the blend when omitted.
        history : pandas.DataFrame or polars.DataFrame, optional
            Earlier raw rows for window algorithms' windows; they are not scored.
        allow_short_history : bool, optional
            Score window algorithms on rows with fewer than ``window - 1`` earlier rows in their group (their windows
            are padded and masked); refused by default.

        Returns
        -------
        pandas.DataFrame
            One column per algorithm (and ``blend``), indexed like ``frame``.
        """
        if not self.classification:
            raise ValueError("This run is not a classification; use predict for its values")
        return self.predict_with_probabilities(frame, algorithms, history, allow_short_history)[1]

    def predict(self, frame, algorithms=None, history=None, allow_short_history=False):
        """Each algorithm's predicted label (classification) or value, and the blend's, indexed like ``frame``."""
        return self.predict_with_probabilities(frame, algorithms, history, allow_short_history)[0]

    def predict_with_probabilities(self, frame, algorithms=None, history=None, allow_short_history=False):
        """(predictions, probabilities) as two DataFrames, from one pass; probabilities is None for a regression."""
        names = self._names(algorithms)
        scored = list(dict.fromkeys([a for a in names if a != BLEND] + (self.blend if BLEND in names else [])))
        raw = self.pipeline.conform(frame)
        X = self.pipeline.transform(self.pipeline.prepare(raw))
        if self.sequence:       # training stacked the tabular columns and the windows as one float32 matrix
            X = np.asarray(X, dtype=np.float32)
            if any(is_window_algo(a) for a in scored):
                X = np.hstack([X, self._windows(raw, history, allow_short_history)])
        preds, probas = {}, {}
        for algo in sorted(scored, key=is_window_algo):     # tabular first, as in training (torch after LightGBM)
            X_algo, _ = algo_matrix(self._selection, algo, X)
            estimator = self.estimator(algo)
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

    def _windows(self, raw, history, allow_short_history):
        spec = self.sequence
        columns = window_columns(spec)
        frame = raw.select(columns).to_pandas()
        past = frame.iloc[:0]
        if history is not None:
            if not isinstance(history, pl.DataFrame):
                history = pl.from_pandas(pd.DataFrame(history).reset_index(drop=True))
            past = history.select(columns).cast({c: self.pipeline.raw_schema[c] for c in columns}).to_pandas()
            keys = [spec["group"], spec["order"]]   # a row in both is the frame's
            seen = pd.MultiIndex.from_frame(frame[keys])
            past = past[~pd.MultiIndex.from_frame(past[keys]).isin(seen)]
        rows = pd.concat([past, frame], ignore_index=True)
        if not allow_short_history:
            refuse_short_history(rows, len(past), spec)
        return window_block(rows, spec)[len(past):]


def refuse_short_history(rows, start, spec):
    """Refuse the rows from position ``start`` on that have fewer than ``window - 1`` earlier rows in their group
    (with a session column, only rows whose session may begin before the rows given), naming how many and where."""
    group, need = spec["group"], int(spec["window"]) - 1
    keys = rows.assign(_key=order_key(rows[spec["order"]].reset_index(drop=True)).to_numpy())
    keys = keys.sort_values([group, "_key"], kind="stable")
    short = keys.groupby(group, sort=False).cumcount() < need
    if spec.get("session"):     # a window never crosses a session
        session = keys[spec["session"]]
        short &= session == session.groupby(keys[group], sort=False).transform("first")
    short = short[short & (short.index >= start)]
    if len(short):
        groups = sorted(map(str, keys.loc[short.index, group].unique()))
        raise ValueError(f"{len(short)} rows (groups {', '.join(groups[:10])}) have fewer than {need} earlier rows "
                         "in their group for their windows. Include earlier rows in the frame, or pass history= "
                         "(history_from_training(frame) when the rows follow training's), or pass "
                         "allow_short_history=True to score padded windows.")


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
    missing = [a for a in manifest["algorithms"] if not (model_dir / "estimators" / f"{a}.pkl").exists()]
    if missing:
        raise FileNotFoundError(f"{run_dir} lacks the saved estimators of {missing}")
    pipeline = joblib.load(pipeline_path)
    feature_map = joblib.load(model_dir / "feature_map.pkl")
    logger.info("Loaded run %s: %s", run_dir, ", ".join(manifest["algorithms"]))
    return TrainedRun(run_dir, manifest, pipeline, feature_map)
