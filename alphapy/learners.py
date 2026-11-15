################################################################################
#
# Package   : AlphaPy
# Module    : learners
# Created   : November 14, 2026
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

"""Classifiers that learn their own features from raw windows (the ``sequence`` section).

``LearnedFeatureClassifier`` takes the window block (AlphaPy gives window algorithms only the block), fits a feature
learner on its training rows only, and fits a classifier on the learned features. Because the learner is fitted inside
``fit``, every cross-validation, calibration and time-ordered fold refits it on that fold's training rows.

Learners (optional dependencies): ``minirocket`` (aeon's MiniRocket random convolution kernels), ``tsfresh``
(tsfresh feature extraction with its FRESH hypothesis-test selection), ``symbolic`` (gplearn's genetic-programming
``SymbolicTransformer`` over the window's values), ``ts2vec`` (alphapy.ts2vec), ``kronos`` (frozen Kronos embeddings,
alphapy.kronos_features) and ``formula`` (reinforcement-learning formula search, alphapy.formula_search).
"""

#
# Imports
#

import logging

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from alphapy.sequence import unflatten

logger = logging.getLogger(__name__)


class _MiniRocket:
    def __init__(self, n_kernels=10000, random_state=0, n_jobs=1):
        from aeon.transformations.collection.convolution_based import MiniRocket
        self.model = MiniRocket(n_kernels=n_kernels, random_state=random_state, n_jobs=n_jobs)

    def fit(self, windows, y):
        self.model.fit(windows)
        return self

    def transform(self, windows):
        return np.asarray(self.model.transform(windows), dtype=np.float32)


class _Tsfresh:
    def __init__(self, fc_parameters="minimal", fdr_level=0.05, n_jobs=1, random_state=0):
        self.fc_parameters, self.fdr_level, self.n_jobs = fc_parameters, fdr_level, n_jobs

    def _extract(self, windows):
        import pandas as pd
        from tsfresh import extract_features
        from tsfresh.feature_extraction import ComprehensiveFCParameters, EfficientFCParameters, MinimalFCParameters

        params = {"minimal": MinimalFCParameters, "efficient": EfficientFCParameters,
                  "comprehensive": ComprehensiveFCParameters}[self.fc_parameters]()
        n, c, w = windows.shape
        long = pd.DataFrame({"id": np.repeat(np.arange(n), w), "time": np.tile(np.arange(w), n),
                             **{f"c{k}": windows[:, k, :].ravel() for k in range(c)}})
        features = extract_features(long, column_id="id", column_sort="time", default_fc_parameters=params,
                                    n_jobs=self.n_jobs, disable_progressbar=True)
        return features.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def fit_transform(self, windows, y):
        from tsfresh import select_features

        features = self._extract(windows)                  # extracted once; the selection reuses it
        selected = select_features(features, np.asarray(y), fdr_level=self.fdr_level, n_jobs=self.n_jobs)
        self.columns_ = list(selected.columns) or list(features.columns[:1])
        return features[self.columns_].to_numpy(dtype=np.float32)

    def transform(self, windows):
        return self._extract(windows).reindex(columns=self.columns_, fill_value=0.0).to_numpy(dtype=np.float32)


class _Symbolic:
    def __init__(self, n_components=10, generations=20, population_size=1000, random_state=0, n_jobs=1, **kwargs):
        from gplearn.genetic import SymbolicTransformer
        self.model = SymbolicTransformer(n_components=n_components, generations=generations,
                                         population_size=population_size, hall_of_fame=min(100, population_size),
                                         random_state=random_state, n_jobs=n_jobs, **kwargs)

    def fit(self, windows, y):
        self.model.fit(windows.reshape(len(windows), -1), np.asarray(y, dtype=float))
        return self

    def transform(self, windows):
        return np.asarray(self.model.transform(windows.reshape(len(windows), -1)), dtype=np.float32)


def _ts2vec(**params):
    from alphapy.ts2vec import TS2VecLearner
    return TS2VecLearner(**params)


def _kronos(**params):
    from alphapy.kronos_features import KronosLearner
    return KronosLearner(**params)


def _formula(**params):
    from alphapy.formula_search import FormulaSearchLearner
    return FormulaSearchLearner(**params)


# Learners that need the window's channel names receive ``channels``; the others do not take it.
LEARNERS = {"minirocket": _MiniRocket, "tsfresh": _Tsfresh, "symbolic": _Symbolic, "ts2vec": _ts2vec,
            "kronos": _kronos, "formula": _formula}
TAKES_CHANNELS = {"kronos", "formula"}


def _classifier(name, params, random_state):
    if name == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=random_state,
                                                                  **(params or {})))
    if name == "lgbm":
        import lightgbm as lgb
        # One thread: a multithreaded LightGBM segfaults in a process where torch was loaded first.
        return lgb.LGBMClassifier(random_state=random_state, verbose=-1, n_jobs=1, **(params or {}))
    raise ValueError(f"Unknown classifier {name}")


class LearnedFeatureClassifier(ClassifierMixin, BaseEstimator):
    """A feature learner on the window block, then a classifier; both fitted on the rows ``fit`` receives.

    ``X`` is the window block alone (AlphaPy gives window algorithms only the block). Learned features are produced
    ``chunk_rows`` rows at a time, and a learner whose features for all rows would exceed ``max_gb`` is refused.
    ``n_jobs`` below 1 means 1: AlphaPy's ``-1`` would oversubscribe the cross-validation workers (numba refuses it,
    tsfresh rejects it).
    """

    def __init__(self, learner="minirocket", n_channels=6, window=20, classifier="logistic", learner_params=None,
                 classifier_params=None, channels=None, chunk_rows=50000, max_gb=8.0, n_jobs=1, random_state=0):
        self.learner = learner
        self.n_channels = n_channels
        self.window = window
        self.classifier = classifier
        self.learner_params = learner_params
        self.classifier_params = classifier_params
        self.channels = channels
        self.chunk_rows = chunk_rows
        self.max_gb = max_gb
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _windows(self, X):
        width = self.n_channels * self.window
        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] != width:
            raise ValueError(f"Expected the {width}-column window block, got {X.shape[1]} columns")
        return unflatten(X, self.n_channels, self.window)

    def _features(self, windows, first=None):
        parts = [] if first is None else [first]
        start = 0 if first is None else len(first)
        for i in range(start, len(windows), self.chunk_rows):
            part = self.learner_.transform(windows[i:i + self.chunk_rows])
            if not parts and part.size and len(windows) * part.shape[1] * 4 / 1e9 > self.max_gb:
                raise MemoryError(f"{self.learner} features for {len(windows)} rows need "
                                  f"{len(windows) * part.shape[1] * 4 / 1e9:.1f} GB, above max_gb={self.max_gb}")
            parts.append(part)
        return np.vstack(parts) if parts else np.zeros((0, 0), np.float32)

    def fit(self, X, y):
        y = np.asarray(y).ravel()
        windows = self._windows(X)
        params = {"n_jobs": max(1, int(self.n_jobs)), **(self.learner_params or {})}
        if self.learner in TAKES_CHANNELS and self.channels is not None:
            params.setdefault("channels", tuple(self.channels))
        self.learner_ = LEARNERS[self.learner](random_state=self.random_state, **params)
        if hasattr(self.learner_, "fit_transform"):
            features = self._features(windows, first=self.learner_.fit_transform(windows, y))
        else:
            self.learner_.fit(windows, y)
            features = self._features(windows)
        self.model_ = _classifier(self.classifier, self.classifier_params, self.random_state)
        self.model_.fit(features, y)
        self.classes_ = self.model_.classes_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(self._features(self._windows(X)))

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
