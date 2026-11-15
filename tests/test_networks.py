"""Sequence networks: each learns a pattern visible only in the window's shape, and training is reproducible."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from alphapy.sequence import flatten

torch = pytest.importorskip("torch")
from alphapy.networks import SequenceNetClassifier  # noqa: E402

WINDOW = 20


def planted(n, seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    shape = 1 - 0.1 * np.sin(np.linspace(0, np.pi, WINDOW))
    close = np.where(y[:, None] == 1, shape, 2 - shape) + 0.04 * rng.normal(size=(n, WINDOW))
    close[:, -1] = 1.0
    windows = np.stack([close, np.ones((n, WINDOW))], axis=1).astype(np.float32)
    return np.hstack([rng.normal(size=(n, 2)).astype(np.float32), flatten(windows)]), y


@pytest.mark.parametrize("arch", ["cnn", "lstm", "gru", "inception", "tcn", "patch", "cnn_lstm"])
def test_each_network_learns_a_pattern_visible_only_in_the_window_shape(arch):
    X, y = planted(800, 0)
    X_new, y_new = planted(400, 1)
    model = SequenceNetClassifier(arch=arch, n_channels=2, window=WINDOW, hidden=32, epochs=25, device="cpu").fit(X, y)
    p = model.predict_proba(X_new)
    assert p.shape == (400, 2) and np.allclose(p.sum(axis=1), 1)
    assert roc_auc_score(y_new, p[:, 1]) > 0.8


def test_training_on_cpu_is_reproducible():
    X, y = planted(300, 2)
    a = SequenceNetClassifier(arch="cnn", n_channels=2, window=WINDOW, hidden=16, epochs=3, device="cpu").fit(X, y)
    b = SequenceNetClassifier(arch="cnn", n_channels=2, window=WINDOW, hidden=16, epochs=3, device="cpu").fit(X, y)
    assert np.allclose(a.predict_proba(X), b.predict_proba(X))


def test_a_fitted_network_pickles_and_predicts_the_same_after_loading():
    import pickle
    X, y = planted(200, 5)
    model = SequenceNetClassifier(arch="gru", n_channels=2, window=WINDOW, hidden=16, epochs=2).fit(X, y)
    loaded = pickle.loads(pickle.dumps(model))
    assert np.allclose(loaded.predict_proba(X), model.predict_proba(X), atol=1e-5)


def test_networks_train_after_lightgbm_in_the_same_process():
    # LightGBM and torch each bundle OpenMP; an LSTM on the CPU deadlocked after a LightGBM fit before torch_threads.
    lgb = pytest.importorskip("lightgbm")
    X, y = planted(200, 6)
    lgb.LGBMClassifier(n_estimators=5, verbose=-1).fit(X, y)
    model = SequenceNetClassifier(arch="lstm", n_channels=2, window=WINDOW, hidden=8, epochs=1, device="cpu")
    assert model.fit(X, y).predict_proba(X).shape == (200, 2)
