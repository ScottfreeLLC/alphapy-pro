"""Chart images drawn as in Jiang, Kelly and Xiu (2023), and the CNN learning a shape from them."""

import numpy as np
import pytest

from alphapy.chart_cnn import image_height, render
from alphapy.sequence import flatten

CHANNELS = ["open", "high", "low", "close", "volume", "_mask"]


def test_each_bar_is_three_pixels_with_open_left_close_right_and_volume_at_the_bottom():
    # Two bars: bar 0 opens at the low and closes at the high; bar 1 is the reverse with half the volume.
    w = np.array([[[10, 12], [12, 12], [10, 10], [12, 10], [100, 50], [1, 1]]], dtype=np.float32)
    w = np.concatenate([np.zeros((1, 6, 3), np.float32), w], axis=2)      # three masked bars first (left black)
    img = render(w, CHANNELS)[0]
    height = image_height(5)
    assert img.shape == (height, 15) and height == 32
    price_rows, vol_rows = height - height // 5 - 1, height // 5
    top, bottom = 0, price_rows - 1                                       # the high and the low price rows
    assert img[:, :9].max() == 0                                           # masked bars are black
    assert img[bottom, 9] == 255 and img[top, 11] == 255                   # bar 0: open at the low, close at the high
    assert img[top, 12] == 255 and img[bottom, 14] == 255                  # bar 1: open high, close low
    assert img[top:bottom + 1, 10].all() and img[top:bottom + 1, 13].all()  # high-low bars span the range
    assert img[height - vol_rows:, 10].all()                               # full volume bar
    assert img[height - vol_rows // 2:, 13].all() and not img[height - vol_rows, 13]   # half volume
    assert img[price_rows, :].max() == 0                                   # the gap row between price and volume


def planted(n, seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    shape = 1 - 0.1 * np.sin(np.linspace(0, np.pi, 20))
    close = np.where(y[:, None] == 1, shape, 2 - shape) + 0.01 * rng.normal(size=(n, 20))
    w = np.stack([close, close * 1.01, close * 0.99, close, np.ones_like(close), np.ones_like(close)], axis=1)
    return flatten(w.astype(np.float32)), y


def test_the_cnn_learns_a_shape_from_the_images():
    pytest.importorskip("torch")
    from sklearn.metrics import roc_auc_score

    from alphapy.chart_cnn import ChartCNNClassifier
    X, y = planted(600, 0)
    X_new, y_new = planted(300, 1)
    model = ChartCNNClassifier(window=20, epochs=8, lr=1e-3, batch_size=64, device="cpu").fit(X, y)
    assert roc_auc_score(y_new, model.predict_proba(X_new)[:, 1]) > 0.8
