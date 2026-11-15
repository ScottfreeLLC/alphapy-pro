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
    # As the pipeline leaves it: LOFO kept a, c, d, e; RFE then kept a, d, e and already shortened the names.
    X = np.arange(20.0).reshape(4, 5)
    model = SimpleNamespace(specs={}, feature_names=list("abcde"), fnames_algo={"LGB": ["a", "d", "e"]},
                            feature_map={("support_lofo", "LGB"): np.array([True, False, True, True, True])},
                            support={"LGB": np.array([True, False, True, True])})
    matrix, names = algo_matrix(model, "LGB", X)
    assert names == ["a", "d", "e"] and np.array_equal(matrix, X[:, [0, 3, 4]])


def test_a_mismatched_rfe_support_or_stale_names_do_not_crash():
    X = np.arange(20.0).reshape(4, 5)
    model = SimpleNamespace(specs={}, feature_names=list("abcdefg"), fnames_algo={}, feature_map={},
                            support={"RF": np.array([True, False, True])})          # RFE fitted on another matrix
    matrix, names = algo_matrix(model, "RF", X)
    assert matrix.shape == (4, 5) and names == ["x0", "x1", "x2", "x3", "x4"]


def test_a_scaled_logistic_pipeline_is_explained_by_its_linear_step(data):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X, y = data
    est = make_pipeline(StandardScaler(), LogisticRegression()).fit(X, y)   # how AlphaPy builds LOGR
    values, unit = explain_estimator(est, X[:100], X[:100])
    assert unit == "log-odds" and ranking(values, list("abcd"))["feature"].iloc[0] == "c"


def test_regressors_are_explained_in_their_predicted_value_and_multiclass_is_refused(data):
    from sklearn.ensemble import RandomForestRegressor
    X, y = data
    _, unit = explain_estimator(RandomForestRegressor(n_estimators=10, random_state=0).fit(X, X[:, 2]), X[:50], X[:50])
    assert unit == "prediction"
    three = RandomForestClassifier(n_estimators=10, random_state=0).fit(X, (X[:, 2] > 0).astype(int) + (X[:, 1] > 1))
    with pytest.raises(ValueError, match="multiclass"):
        explain_estimator(three, X[:20], X[:20])


def test_low_variance_removal_keeps_the_feature_names_with_the_columns():
    from alphapy.features import remove_lv_features
    X = np.column_stack([np.arange(6.0), np.ones(6), np.arange(6.0) ** 2])          # the middle column is constant
    model = SimpleNamespace(specs={"lv_remove": True, "lv_threshold": 0.0, "predict_mode": False},
                            feature_names=["a", "b", "c"], feature_map={}, algolist=["LGB"], fnames_algo={})
    reduced = remove_lv_features(model, X)
    assert reduced.shape[1] == 2 and model.feature_names == ["a", "c"] and model.fnames_algo["LGB"] == ["a", "c"]


def test_window_models_are_explained_by_channel_and_bar_age_and_the_values_add_up():
    from alphapy.explain import explain_window_model, window_groups
    window, channels = 12, ["close", "volume", "_mask"]
    names, columns = window_groups(channels, window)
    assert names[:4] == ["close[own bar]", "close[2-5 back]", "close[6-10 back]", "close[11-12 back]"]
    assert list(columns[0]) == [window - 1] and "_mask" not in " ".join(names)

    class SixToTen:                                    # depends only on close, 6-10 bars back
        def predict_proba(self, X):
            p = 1 / (1 + np.exp(-X[:, columns[2]].sum(axis=1)))
            return np.column_stack([1 - p, p])

    rng = np.random.default_rng(0)
    X, back = rng.normal(size=(30, 3 * window)), rng.normal(size=(20, 3 * window))
    values, got = explain_window_model(SixToTen(), X, back, channels, window)
    assert got == names and values.shape == (30, len(names))
    assert np.abs(values).mean(axis=0).argmax() == 2
    f = SixToTen().predict_proba
    assert np.allclose(values.sum(axis=1), f(X)[:, 1] - f(back)[:, 1].mean(), atol=1e-6)
