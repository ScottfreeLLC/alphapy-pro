"""SHAP explanations: the feature that drives the label ranks first, each explainer reports its unit, and each
algorithm is explained on the columns it was fitted on."""

from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from alphapy.explain import explain_estimator, ranking, sample_rows
from alphapy.model import algo_matrix

pytest.importorskip("shap")


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1500, 4))
    y = (X[:, 2] + 0.3 * rng.normal(size=1500) > 0).astype(int)   # only the third column matters
    return X, y


@pytest.mark.parametrize("estimator, unit", [
    (LogisticRegression(), "log-odds"),
    (RandomForestClassifier(n_estimators=30, max_depth=4, random_state=0), "probability"),
    (KNeighborsClassifier(n_neighbors=25), "probability"),             # the permutation explainer
])
def test_the_driving_feature_ranks_first_and_the_unit_is_reported(data, estimator, unit):
    X, y = data
    estimator.fit(X, y)
    values, got_unit = explain_estimator(estimator, X[:200], X[:100], seed=1)
    assert values.shape == (200, 4) and got_unit == unit
    assert ranking(values, ["a", "b", "c", "d"])["feature"].iloc[0] == "c"


def test_boosted_trees_are_explained_in_log_odds(data):
    lgb = pytest.importorskip("lightgbm")
    X, y = data
    est = lgb.LGBMClassifier(n_estimators=50, verbose=-1).fit(X, y)
    values, unit = explain_estimator(est, X[:100], X[:50])
    assert unit == "log-odds" and values.shape == (100, 4)
    assert ranking(values, list("abcd"))["feature"].iloc[0] == "c"


def test_the_sample_is_seeded_sorted_and_whole_when_small():
    one, two = sample_rows(10_000, 50, 7), sample_rows(10_000, 50, 7)
    assert np.array_equal(one, two) and np.all(np.diff(one) > 0) and len(one) == 50
    assert np.array_equal(sample_rows(30, 50, 7), np.arange(30))


def test_an_algorithm_is_explained_on_the_columns_it_was_fitted_on():
    X = np.arange(20.0).reshape(4, 5)
    model = SimpleNamespace(feature_names=list("abcde"), fnames_algo={"LGB": ["a", "c", "d", "e"]},
                            feature_map={("support_lofo", "LGB"): np.array([True, False, True, True, True])},
                            support={"LGB": np.array([True, False, True, True])})
    matrix, names = algo_matrix(model, "LGB", X)
    assert names == ["a", "d", "e"] and np.array_equal(matrix, X[:, [0, 3, 4]])
