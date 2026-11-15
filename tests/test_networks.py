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
    return flatten(windows), y


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
    # AlphaPy's order: LightGBM is imported and fitted first, then a network trains in the same process. An LSTM on
    # the CPU deadlocked there before torch_threads. Run in a fresh process so the import order is AlphaPy's, not the
    # test session's (torch imported before LightGBM makes multithreaded LightGBM segfault).
    import subprocess
    import sys
    pytest.importorskip("lightgbm")
    script = """
import numpy as np
import lightgbm as lgb
from alphapy.networks import SequenceNetClassifier
rng = np.random.default_rng(6)
X, y = rng.normal(size=(200, 40)).astype(np.float32), rng.integers(0, 2, 200)
lgb.LGBMClassifier(n_estimators=5, verbose=-1).fit(X, y)
p = SequenceNetClassifier(arch="lstm", n_channels=2, window=20, hidden=8, epochs=1, device="cpu").fit(X, y).predict_proba(X)
assert p.shape == (200, 2)
"""
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr[-2000:]


def test_padding_is_not_standardised_as_data_and_the_mask_is_left_alone():
    from alphapy.networks import standardise, standardise_stats
    w = np.ones((4, 2, 5), np.float32)
    w[:, 0, :] = np.arange(5) + 10.0
    w[0, :, :2] = 0.0                                      # row 0: two padded bars (data 0, mask 0)
    mean, std = standardise_stats(w)
    assert np.isclose(mean[0, 0], (4 * 12 * 5 - 21) / 18)  # present bars only
    out = standardise(w, mean, std)
    assert np.all(out[0, 0, :2] == 0) and np.array_equal(out[:, 1], w[:, 1])
    # A short row at predict time when no fitted row was short stays bounded.
    full = np.ones((10, 2, 5), np.float32)
    m2, s2 = standardise_stats(full)
    short = full[:1].copy()
    short[0, :, :3] = 0
    assert np.abs(standardise(short, m2, s2)).max() < 10


def test_labels_other_than_zero_and_one_and_threads_restored():
    X, y = planted(200, 7)
    before = torch.get_num_threads()
    model = SequenceNetClassifier(arch="cnn", n_channels=2, window=WINDOW, hidden=16, epochs=2, device="cpu")
    model.fit(X, np.where(y == 1, 1, -1))
    assert set(model.predict(X)) <= {-1, 1} and list(model.classes_) == [-1, 1]
    assert torch.get_num_threads() == before
    with pytest.raises(ValueError, match="multiple of 4"):
        SequenceNetClassifier(arch="patch", n_channels=2, window=WINDOW, hidden=30, device="cpu").fit(X, y)
