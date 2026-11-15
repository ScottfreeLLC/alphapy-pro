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

"""SHAP explanations of fitted estimators (optional dependency ``shap``).

Each algorithm is explained on a seeded sample of the test rows (the training rows when there is no test partition),
using the estimator fitted on all training rows before any calibration. Tree ensembles use SHAP's exact tree
explainer, linear models its linear explainer, and anything else a permutation explainer on the predicted probability.
Units follow the explainer: log-odds for boosted trees and linear models, probability for random forests and the
permutation explainer; the unit is written into each output.
"""

#
# Imports
#

import logging

import numpy as np
import pandas as pd

from alphapy.globals import SSEP, USEP, Partition, datasets

logger = logging.getLogger(__name__)

TREE_TYPES = ("LGBM", "XGB", "CatBoost", "RandomForest", "ExtraTrees", "GradientBoosting", "DecisionTree")
LINEAR_TYPES = ("LogisticRegression", "LinearRegression", "Ridge", "Lasso", "LinearSVC")
PROBABILITY_TREES = ("RandomForest", "ExtraTrees", "DecisionTree")


def _kind(estimator) -> str:
    name = type(estimator).__name__
    if any(t in name for t in TREE_TYPES):
        return "tree"
    if any(t in name for t in LINEAR_TYPES):
        return "linear"
    return "other"


def explain_estimator(estimator, X: np.ndarray, background: np.ndarray, seed: int = 42) -> tuple[np.ndarray, str]:
    """SHAP values for the positive class (rows x features) and their unit."""
    import shap

    kind, name = _kind(estimator), type(estimator).__name__
    if kind == "tree":
        explainer = shap.TreeExplainer(estimator)
        values = explainer.shap_values(X)
        unit = "probability" if any(t in name for t in PROBABILITY_TREES) else "log-odds"
    elif kind == "linear":
        explainer = shap.LinearExplainer(estimator, shap.maskers.Independent(background))
        values = explainer.shap_values(X)
        unit = "log-odds"
    else:
        positive = (lambda x: estimator.predict_proba(x)[:, 1]) if hasattr(estimator, "predict_proba") \
            else estimator.predict
        explainer = shap.PermutationExplainer(positive, shap.maskers.Independent(background), seed=seed)
        values = explainer(X).values
        unit = "probability" if hasattr(estimator, "predict_proba") else "prediction"
    if isinstance(values, list):      # one array per class: keep the positive class
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:              # rows x features x classes
        values = values[..., -1]
    return values, unit


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


def shap_analysis(model, partition: Partition) -> dict:
    """Explain every fitted algorithm on a sample of ``partition`` rows; write plots and tables to the run."""
    import matplotlib.pyplot as plt
    import shap

    from alphapy.model import algo_matrix
    from alphapy.plots import get_plot_directory, write_plot

    size, seed = model.specs["shap_samples"], model.specs["seed"]
    X_all = model.X_test if partition == Partition.test else model.X_train
    out_dir = SSEP.join([model.specs["run_dir"], "output"])
    plot_dir, pstring = get_plot_directory(model), datasets[partition]
    summary = {}
    for algo in model.algolist:
        estimator = model.uncalibrated.get(algo, model.estimators.get(algo))
        if estimator is None:
            continue
        X, names = algo_matrix(model, algo, X_all)
        background, _ = algo_matrix(model, algo, model.X_train)
        rows = sample_rows(X.shape[0], size, seed)
        back = background[sample_rows(background.shape[0], min(size, 500), seed + 1)]
        logger.info("SHAP for Algorithm: %s (%d rows)", algo, len(rows))
        try:
            values, unit = explain_estimator(estimator, X[rows], back, seed)
        except Exception as e:  # noqa: BLE001 — one algorithm's failure must not stop the run
            logger.info("SHAP failed for %s: %s", algo, e)
            continue
        tag = USEP.join([pstring, algo])
        table = ranking(values, names)
        table.to_csv(SSEP.join([out_dir, f"shap_ranking_{tag}.csv"]), index=False)
        pd.DataFrame(values, columns=names).assign(unit=unit).to_csv(
            SSEP.join([out_dir, f"shap_values_{tag}.csv"]), index=False)
        explanation = shap.Explanation(values=values, data=X[rows], feature_names=list(names))
        for kind, draw in (("shap_summary", shap.plots.beeswarm), ("shap_bar", shap.plots.bar)):
            plt.figure()
            draw(explanation, max_display=20, show=False)
            plt.title(f"{algo} SHAP [{pstring}, {unit}, before calibration]")
            write_plot("matplotlib", plt, kind, tag, plot_dir)
        summary[algo] = {"unit": unit, "rows": len(rows), "top": table["feature"].head(5).tolist()}
    return summary
