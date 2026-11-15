"""Learned-feature classifiers: each finds a pattern visible only in the window's shape, is fitted only on the rows
it is given (so every fold refits it), and transforms each row on its own."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_score

from alphapy import learners
from alphapy.learners import LearnedFeatureClassifier
from alphapy.sequence import flatten

WINDOW = 20


def planted(n, seed):
    """A dip in the middle (label 1) or a hump (label 0), ending at the same price: the window block alone."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    shape = 1 - 0.1 * np.sin(np.linspace(0, np.pi, WINDOW))          # dip, ends at 1
    close = np.where(y[:, None] == 1, shape, 2 - shape) + 0.04 * rng.normal(size=(n, WINDOW))
    close[:, -1] = 1.0
    windows = np.stack([close, np.ones((n, WINDOW))], axis=1).astype(np.float32)   # close and the mask channel
    return flatten(windows), y


@pytest.mark.parametrize("learner, params", [
    ("minirocket", {"n_kernels": 500}),
    ("tsfresh", {"fc_parameters": "minimal"}),
    ("symbolic", {"n_components": 3, "generations": 5, "population_size": 300}),
])
def test_each_learner_finds_a_pattern_visible_only_in_the_window_shape(learner, params):
    pytest.importorskip({"minirocket": "aeon", "tsfresh": "tsfresh", "symbolic": "gplearn"}[learner])
    X, y = planted(600, 0)
    X_new, y_new = planted(400, 1)
    model = LearnedFeatureClassifier(learner=learner, n_channels=2, window=WINDOW, learner_params=params).fit(X, y)
    assert roc_auc_score(y_new, model.predict_proba(X_new)[:, 1]) > 0.8
    assert abs(roc_auc_score(y_new, X_new[:, WINDOW - 1]) - 0.5) < 0.1      # the last close carries nothing


class _Spy:
    fitted_rows = []

    def __init__(self, random_state=0, **kwargs):
        pass

    def fit(self, windows, y):
        _Spy.fitted_rows.append(len(windows))
        return self

    def transform(self, windows):
        return windows[:, 0, :]


def test_the_learner_is_fitted_only_on_each_folds_training_rows(monkeypatch):
    monkeypatch.setitem(learners.LEARNERS, "spy", _Spy)
    _Spy.fitted_rows = []
    X, y = planted(300, 2)
    cross_val_score(LearnedFeatureClassifier(learner="spy", n_channels=2, window=WINDOW), X, y, cv=KFold(3))
    assert _Spy.fitted_rows == [200, 200, 200]


def test_a_rows_learned_features_do_not_depend_on_other_rows():
    pytest.importorskip("aeon")
    X, y = planted(200, 3)
    model = LearnedFeatureClassifier(learner="minirocket", n_channels=2, window=WINDOW,
                                     learner_params={"n_kernels": 200}).fit(X, y)
    alone = model.predict_proba(X[:1])
    changed = X.copy()
    changed[1:] = planted(199, 4)[0]
    assert np.allclose(model.predict_proba(changed)[:1], alone)


def test_features_too_large_for_memory_are_refused_and_n_jobs_below_one_means_one(monkeypatch):
    monkeypatch.setitem(learners.LEARNERS, "spy", _Spy)
    X, y = planted(100, 7)
    with pytest.raises(MemoryError, match="max_gb"):
        LearnedFeatureClassifier(learner="spy", n_channels=2, window=WINDOW, max_gb=1e-9).fit(X, y)
    seen = {}

    class Jobs(_Spy):
        def __init__(self, random_state=0, n_jobs=None, **kwargs):
            seen["n_jobs"] = n_jobs
    monkeypatch.setitem(learners.LEARNERS, "jobs", Jobs)
    LearnedFeatureClassifier(learner="jobs", n_channels=2, window=WINDOW, n_jobs=-1).fit(X, y)
    assert seen["n_jobs"] == 1


def test_a_matrix_wider_than_the_window_block_is_refused():
    X, y = planted(50, 8)
    with pytest.raises(ValueError, match="window block"):
        LearnedFeatureClassifier(learner="minirocket", n_channels=2, window=WINDOW).fit(np.hstack([X, X[:, :3]]), y)
