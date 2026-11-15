"""Kronos features: masked bars are ignored, each row is embedded on its own, and the frozen embeddings expose a shape.

Skipped when the optional dependencies are missing or the published weights cannot be downloaded.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("einops")
pytest.importorskip("safetensors")
hub_errors = pytest.importorskip("huggingface_hub.errors")
from alphapy.kronos_features import KronosLearner  # noqa: E402

WINDOW = 32


def _network_errors():
    errors = [hub_errors.LocalEntryNotFoundError, hub_errors.OfflineModeIsEnabled, ConnectionError, TimeoutError]
    try:
        import httpx
        errors.append(httpx.TransportError)
    except ImportError:
        pass
    return tuple(errors)


@pytest.fixture(scope="module")
def learner():
    try:
        return KronosLearner(device="cpu").fit()
    except _network_errors() as e:
        pytest.skip(f"Kronos weights unavailable: {e}")


def candles(close, rng):
    """Windows as the sequence section builds them: open, high, low, close, volume, mask; prices / last close."""
    n, w = close.shape[0], close.shape[1] - 1
    open_ = close[:, :-1] * (1 + 0.003 * rng.normal(size=(n, w)))
    close = close[:, 1:]
    high = np.maximum(open_, close) * (1 + np.abs(0.004 * rng.normal(size=(n, w))))
    low = np.minimum(open_, close) * (1 - np.abs(0.004 * rng.normal(size=(n, w))))
    volume = rng.lognormal(10, 0.3, size=(n, w))
    last = close[:, -1:]
    return np.stack([open_ / last, high / last, low / last, close / last, volume / volume.mean(1, keepdims=True),
                     np.ones((n, w))], axis=1).astype(np.float32)


def random_windows(n, seed):
    rng = np.random.default_rng(seed)
    return candles(100 * np.exp(0.01 * rng.normal(size=(n, WINDOW + 1)).cumsum(axis=1)), rng)


def planted(n, seed, amplitude=0.02):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    shape = amplitude * np.sin(np.linspace(0, np.pi, WINDOW + 1))           # a hump (y = 0) or a dip (y = 1)
    close = 100 * (1 + np.where(y[:, None] == 1, -shape, shape)) * np.exp(0.01 * rng.normal(size=(n, WINDOW + 1)))
    return candles(close, rng), y


def test_short_history_is_masked_out(learner):
    w = random_windows(6, 0)
    w[:, 5, :12] = 0.0                                   # 12 bars of missing history, zero-filled as sequence does
    w[:, :5, :12] = 0.0
    features = learner.transform(w)
    assert features.dtype == np.float32 and features.shape == (6, 512) and np.isfinite(features).all()
    garbage = w.copy()
    garbage[:, :5, :12] = np.random.default_rng(1).normal(50, 30, size=(6, 5, 12))
    assert np.allclose(learner.transform(garbage), features, atol=1e-5)
    alone = learner.transform(np.ascontiguousarray(w[:, :, 12:]))   # the same 20 bars as a 20-bar window
    assert np.allclose(alone, features, atol=1e-4)
    full = random_windows(6, 4)
    capped = KronosLearner(device="cpu", max_context=20).fit().transform(full)   # keeps the last 20 bars
    assert np.allclose(capped, learner.transform(np.ascontiguousarray(full[:, :, -20:])), atol=1e-4)
    empty = w[:1].copy()
    empty[:, 5, :] = 0.0
    assert np.array_equal(learner.transform(empty), np.zeros((1, 512), dtype=np.float32))


def test_each_row_is_embedded_on_its_own(learner):
    w = random_windows(40, 2)
    together = learner.transform(w)
    others = random_windows(40, 3)
    others[7] = w[7]
    assert np.allclose(learner.transform(others)[7], together[7], atol=1e-5)
    assert np.allclose(learner.transform(w[7:8])[0], together[7], atol=1e-5)


def test_frozen_features_expose_a_planted_shape(learner):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    w, y = planted(600, 0)
    w_new, y_new = planted(300, 1)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(learner.transform(w), y)
    assert roc_auc_score(y_new, clf.predict_proba(learner.transform(w_new))[:, 1]) > 0.7
