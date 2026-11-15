################################################################################
#
# Package   : AlphaPy
# Module    : explain
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

"""SHAP explanations of fitted estimators (optional dependency ``shap``, the ``explain`` extra).

Each algorithm is explained on a seeded sample of the test rows (the training rows when there are no test labels),
using the estimator AlphaPy fitted before any calibration (for XGBoost with early stopping, the fit on its training
split). A pipeline (AlphaPy scales linear models and neighbours in one) is unwrapped: its last step is explained on the
output of the steps before it. Tree ensembles use SHAP's exact tree explainer, linear models its linear explainer, and
anything else a permutation explainer on fewer rows. Binary classifiers and regressors are supported; multiclass
models are skipped with a message. Window models (the ``sequence`` section) are explained by channel and bar age:
the row's own bar, 2-5 bars back, 6-10, 11-20, 21-60 and older, each group's columns masked together.

Units: log-odds for boosted trees and linear classifiers, the probability of the positive class for random forests,
extra trees and the permutation explainer, and the predicted value for regressors.
"""

#
# Imports
#

import logging

import numpy as np
import pandas as pd
from sklearn.base import is_classifier
from sklearn.pipeline import Pipeline

from alphapy.globals import SSEP, USEP, Partition, datasets

logger = logging.getLogger(__name__)

TREE_TYPES = ("LGBM", "XGB", "CatBoost", "RandomForest", "ExtraTrees", "GradientBoosting", "DecisionTree")
LINEAR_TYPES = ("LogisticRegression", "LinearRegression", "LinearSVC", "LinearSVR")
PROBABILITY_TREES = ("RandomForest", "ExtraTrees", "DecisionTree")
BACKGROUND_ROWS = 100
PERMUTATION_ROWS = 200


def unwrap(estimator, X: np.ndarray, background: np.ndarray):
    """A pipeline's last step and both matrices passed through the steps before it; anything else unchanged."""
    if isinstance(estimator, Pipeline) and len(estimator.steps) > 1:
        head = estimator[:-1]
        return estimator[-1], head.transform(X), head.transform(background)
    return estimator, X, background


def _kind(estimator) -> str:
    name = type(estimator).__name__
    if any(t in name for t in TREE_TYPES):
        return "tree"
    if any(t in name for t in LINEAR_TYPES):
        return "linear"
    return "other"


def _unit(estimator, kind: str) -> str:
    if not is_classifier(estimator):
        return "prediction"
    if kind == "tree":
        return "probability" if any(t in type(estimator).__name__ for t in PROBABILITY_TREES) else "log-odds"
    return "log-odds" if kind == "linear" else "probability"


def explain_estimator(estimator, X: np.ndarray, background: np.ndarray, seed: int = 42) -> tuple[np.ndarray, str]:
    """SHAP values (rows x features: the positive class of a binary classifier, or a regressor's prediction), unit."""
    import shap

    estimator, X, background = unwrap(estimator, X, background)
    if is_classifier(estimator) and len(getattr(estimator, "classes_", [0, 1])) > 2:
        raise ValueError("multiclass models are not explained")
    kind = _kind(estimator)
    masker = shap.maskers.Independent(background, max_samples=len(background))
    if kind == "tree":
        values = shap.TreeExplainer(estimator).shap_values(X)
    elif kind == "linear":
        values = shap.LinearExplainer(estimator, masker).shap_values(X)
    else:
        predict = (lambda x: estimator.predict_proba(x)[:, 1]) if is_classifier(estimator) else estimator.predict
        explainer = shap.PermutationExplainer(predict, masker, seed=seed)
        values = explainer(X, max_evals=max(500, 2 * X.shape[1] + 1)).values
    values = np.asarray(values)
    if values.ndim == 3:              # rows x features x classes: the positive class
        values = values[..., -1]
    return values, _unit(estimator, kind)


AGE_GROUPS = ((0, 1), (1, 5), (5, 10), (10, 20), (20, 60), (60, None))   # bars back from the row's own bar


def window_groups(channels: list[str], window: int) -> tuple[list[str], list[np.ndarray]]:
    """Names and column positions (in the flattened block, channel by channel, oldest bar first) of each channel's
    bar-age groups: the row's own bar, bars 2-5 back, 6-10, 11-20, 21-60, older. The mask channel is not explained."""
    names, columns = [], []
    for c, channel in enumerate(channels):
        if channel == "_mask":
            continue
        for lo, hi in AGE_GROUPS:
            hi = window if hi is None else min(hi, window)
            if lo >= hi:
                continue
            ages = np.arange(lo, hi)                        # 0 = the row's own bar
            names.append(f"{channel}[{'own bar' if hi - lo == 1 and lo == 0 else f'{lo + 1}-{hi} back'}]")
            columns.append(c * window + (window - 1 - ages))
    return names, columns


def explain_window_model(estimator, X: np.ndarray, background: np.ndarray, channels: list[str], window: int,
                         seed: int = 42) -> tuple[np.ndarray, list[str]]:
    """Interventional SHAP values of a window model's probability, one per channel and bar-age group.

    Each explained row is coded as a vector of row numbers, one per group; masking a group takes its columns from a
    background row instead. SHAP's permutation explainer runs on that coding, so the values are exact Shapley
    estimates over groups and sum to the prediction minus the background's average.
    """
    import shap

    names, columns = window_groups(channels, window)
    data = np.vstack([X, background])
    own = np.tile(np.arange(len(X))[:, None], (1, len(columns)))
    back = np.tile(np.arange(len(X), len(data))[:, None], (1, len(columns)))

    def predict(codes):
        codes = np.asarray(codes, dtype=int)
        rebuilt = np.repeat(data[:1], len(codes), axis=0)
        rebuilt[:] = data[codes[:, 0]]                      # columns outside every group follow the first group's row
        for g, cols in enumerate(columns):
            rebuilt[:, cols] = data[codes[:, g]][:, cols]
        return estimator.predict_proba(rebuilt)[:, 1]

    explainer = shap.PermutationExplainer(predict, shap.maskers.Independent(back, max_samples=len(back)), seed=seed)
    values = explainer(own, max_evals=max(500, 2 * len(columns) + 1)).values
    return np.asarray(values), names


def sample_rows(n_rows: int, size: int, seed: int) -> np.ndarray:
    """A sorted seeded sample of row positions (all rows when there are fewer than ``size``)."""
    if n_rows <= size:
        return np.arange(n_rows)
    return np.sort(np.random.default_rng(seed).choice(n_rows, size, replace=False))


def ranking(values: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Features by mean absolute SHAP value, largest first."""
    mean_abs = np.abs(values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    return pd.DataFrame({"feature": np.asarray(names)[order], "mean_abs_shap": mean_abs[order]})


def shap_analysis(model, partition: Partition) -> None:
    """Explain every fitted algorithm on a sample of ``partition`` rows; write plots and tables to the run. A failure
    is logged and never stops the run."""
    try:
        import matplotlib.pyplot as plt
        import shap
    except ImportError:
        logger.warning("SHAP needs the explain extra (pip install alphapy-pro[explain]); skipped")
        return

    from alphapy.model import algo_matrix
    from alphapy.plots import get_plot_directory, write_plot

    size, seed = model.specs["shap_samples"], model.specs["seed"]
    X_all = model.X_test if partition == Partition.test else model.X_train
    out_dir = SSEP.join([model.specs["run_dir"], "output"])
    plot_dir, pstring = get_plot_directory(model), datasets[partition]
    for algo in model.algolist:
        try:
            estimator = model.uncalibrated.get(algo, model.estimators.get(algo))
            if estimator is None:
                continue
            X, names = algo_matrix(model, algo, X_all)
            background, _ = algo_matrix(model, algo, model.X_train)
            window_model = hasattr(estimator, "n_channels") and hasattr(estimator, "window")
            kind = "other" if window_model else _kind(unwrap(estimator, X[:1], X[:1])[0])
            rows = sample_rows(X.shape[0], size if kind != "other" else min(size, PERMUTATION_ROWS), seed)
            back = background[sample_rows(background.shape[0], BACKGROUND_ROWS, seed + 1)]
            logger.info("SHAP for Algorithm: %s (%d rows)", algo, len(rows))
            if window_model:
                from alphapy.sequence import MASK
                channels = list(model.specs["sequence"]["channels"]) + [MASK]
                values, names = explain_window_model(estimator, X[rows], back, channels, estimator.window, seed)
                unit, X = "probability", np.zeros((X.shape[0], len(names)))   # plots colour by value: none here
            else:
                values, unit = explain_estimator(estimator, X[rows], back, seed)
            tag = USEP.join([pstring, algo])
            ranking(values, names).assign(unit=unit).to_csv(SSEP.join([out_dir, f"shap_ranking_{tag}.csv"]),
                                                            index=False)
            pd.DataFrame(values, columns=names).to_csv(SSEP.join([out_dir, f"shap_values_{tag}.csv"]), index=False)
            note = ", before calibration" if algo in model.uncalibrated else ""
            explanation = shap.Explanation(values=values, data=X[rows], feature_names=list(names))
            for plot_name, draw in (("shap_summary", shap.plots.beeswarm), ("shap_bar", shap.plots.bar)):
                plt.figure()
                draw(explanation, max_display=20, show=False)
                plt.title(f"{algo} SHAP [{pstring}, {unit}{note}]")
                write_plot("matplotlib", plt, plot_name, tag, plot_dir)
        except Exception as e:  # noqa: BLE001 — one algorithm's failure must not stop the run
            logger.warning("SHAP failed for %s: %s", algo, e)
