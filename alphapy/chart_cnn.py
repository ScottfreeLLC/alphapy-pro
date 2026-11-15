################################################################################
#
# Package   : AlphaPy
# Module    : chart_cnn
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

"""A CNN on price-volume chart images, after Jiang, Kelly and Xiu, "(Re-)Imag(in)ing Price Trends", Journal of
Finance 78(6), 2023 (optional dependency ``torch``).

Each window is drawn as a black-and-white image, three pixels per bar: the open as a tick on the left, the high-low
range as a vertical bar in the middle, the close as a tick on the right, a moving average of the closes as a line, and
volume as bars in the bottom fifth. The image height follows the paper (32 pixels for 5 bars, 64 for 20, 96 for 60;
other lengths are interpolated). Prices are scaled to the window's own high and low, volume to its own maximum. The
moving average is taken within the window (expanding at its start), since the window holds no earlier bars.

The network follows the paper: blocks of a 5x3 convolution (64 channels, doubling per block; the first with the
paper's vertical stride and dilation), batch normalisation, leaky ReLU and 2x1 max pooling (two blocks for up to 5
bars, three up to 20, four beyond), then 50% dropout and a linear output. Cost: about 1 ms per row per epoch on MPS
at 20 bars, so hours per fit on millions of rows. Images are drawn batch by batch, so the full set never sits in memory.
"""

#
# Imports
#

import logging

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from alphapy.sequence import unflatten

logger = logging.getLogger(__name__)

PRICE_CHANNELS = ("open", "high", "low", "close")


def image_height(window: int) -> int:
    """The paper's heights (5 bars: 32, 20: 64, 60: 96), interpolated for other lengths."""
    return int(round(np.interp(window, [5, 20, 60], [32, 64, 96])))


def render(windows: np.ndarray, channels: list[str]) -> np.ndarray:
    """Chart images (rows, height, 3 * window) as uint8 0/255, the newest bar on the right.

    ``windows`` is (rows, channels, window) from ``alphapy.sequence``; ``channels`` names its rows (the mask last).
    Bars masked out (short history) are left black.
    """
    index = {c: i for i, c in enumerate(channels)}
    n, _, w = windows.shape
    height = image_height(w)
    price_rows = height - height // 5 - 1                       # the bottom fifth holds volume, one row gap
    op, hi_, lo_, cl = (windows[:, index[k], :] for k in PRICE_CHANNELS)
    present = windows[:, -1, :] > 0
    volume = windows[:, index["volume"], :] if "volume" in index else np.zeros_like(cl)
    with np.errstate(invalid="ignore", divide="ignore"):
        hi = np.where(present, hi_, -np.inf).max(axis=1, keepdims=True)
        lo = np.where(present, lo_, np.inf).min(axis=1, keepdims=True)
        span = np.where(hi > lo, hi - lo, 1.0)
        level = lambda p: np.clip(np.round((p - lo) / span * (price_rows - 1)), 0, price_rows - 1).astype(int)  # noqa: E731
        count = np.cumsum(present, axis=1)
        moving = np.cumsum(np.where(present, cl, 0.0), axis=1) / np.maximum(count, 1)
        vmax = np.where(present, volume, 0.0).max(axis=1, keepdims=True)
        vol_rows = np.where(vmax > 0, np.round(volume / np.where(vmax > 0, vmax, 1) * (height // 5)), 0).astype(int)
    rows = np.arange(n)[:, None]
    images = np.zeros((n, height, 3 * w), dtype=np.uint8)
    top = price_rows - 1                                        # image row 0 is the top: price level k is row top-k
    ol, hl, ll, cl_, ml = level(op), level(hi_), level(lo_), level(cl), level(moving)
    for t in range(w):
        on = present[:, t]
        r = rows[on, 0]
        images[r, top - ol[on, t], 3 * t] = 255
        images[r, top - cl_[on, t], 3 * t + 2] = 255
        images[r, top - ml[on, t], 3 * t + 1] = 255
        for k in range(price_rows):                              # the high-low bar
            inside = on & (ll[:, t] <= k) & (k <= hl[:, t])
            images[rows[inside, 0], top - k, 3 * t + 1] = 255
        for k in range(height // 5):                             # the volume bar, from the bottom
            tall = on & (vol_rows[:, t] > k)
            images[rows[tall, 0], height - 1 - k, 3 * t + 1] = 255
    return images


def first_layer(window: int) -> tuple[int, int, int]:
    """(vertical stride, vertical dilation, vertical padding) of the first convolution. The paper uses strides 1, 3, 3
    and dilations 1, 2, 3 for 5-, 20- and 60-bar images, on the first layer only; it does not state the padding. These
    paddings reproduce its fully connected sizes exactly (15,360, 46,080 and 184,320)."""
    if window <= 5:
        return 1, 1, 2
    return (3, 2, 12) if window <= 20 else (3, 3, 12)


def _network(window, height):
    import torch
    from torch import nn

    blocks = 2 if window <= 5 else 3 if window <= 20 else 4
    stride, dilation, padding = first_layer(window)
    layers, c_in = [], 1
    for b in range(blocks):
        c_out = 64 * 2 ** b
        conv = (nn.Conv2d(c_in, c_out, (5, 3), stride=(stride, 1), dilation=(dilation, 1), padding=(padding, 1))
                if b == 0 else nn.Conv2d(c_in, c_out, (5, 3), padding=(2, 1)))
        layers += [conv, nn.BatchNorm2d(c_out), nn.LeakyReLU(0.01), nn.MaxPool2d((2, 1))]
        c_in = c_out
    convs = nn.Sequential(*layers, nn.Flatten())
    from alphapy.networks import torch_threads
    with torch.no_grad(), torch_threads(1):     # any torch CPU work keeps to one thread (alphapy.networks)
        n_flat = convs.eval()(torch.zeros(2, 1, height, 3 * window)).shape[1]
    body = nn.Sequential(convs, nn.Dropout(0.5), nn.Linear(n_flat, 1))
    with torch_threads(1):
        for m in body.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
    return body


class ChartCNNClassifier(ClassifierMixin, BaseEstimator):
    """The chart-image CNN, trained on the rows ``fit`` receives (the latest ``validation`` share for early stopping)."""

    def __init__(self, n_channels=6, window=20, channels=("open", "high", "low", "close", "volume", "_mask"),
                 epochs=30, batch_size=128, lr=1e-5, validation=0.1, patience=2, min_epochs=5, device="auto",
                 torch_threads=1, random_state=0):
        self.n_channels = n_channels
        self.window = window
        self.channels = channels
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.validation = validation
        self.patience = patience
        self.min_epochs = min_epochs
        self.device = device
        self.torch_threads = torch_threads
        self.random_state = random_state

    def _windows(self, X):
        width = self.n_channels * self.window
        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] != width:
            raise ValueError(f"Expected the {width}-column window block, got {X.shape[1]} columns")
        return unflatten(X, self.n_channels, self.window)

    def _images(self, windows):
        import torch
        return torch.from_numpy(render(windows, list(self.channels))).float().div_(255.0).unsqueeze(1)

    def _logits(self, windows):
        import torch

        self.net_.eval()
        with torch.no_grad():
            return torch.cat([self.net_(self._images(windows[i:i + 2048]).to(self.device_)).squeeze(-1).cpu()
                              for i in range(0, len(windows), 2048)])

    def _recalibrate(self, windows, rows=20000):
        """Batch-normalisation statistics recomputed on up to ``rows`` training windows with the current weights; the
        running averages lag behind the weights and make evaluation-mode outputs wrong (huge logits)."""
        import torch

        norms = [m for m in self.net_.modules() if isinstance(m, torch.nn.BatchNorm2d)]
        for m in norms:
            m.reset_running_stats()
            m.momentum = None                          # a cumulative average over the pass
        sample = windows[np.random.default_rng(self.random_state).permutation(len(windows))[:rows]]
        self.net_.train()
        with torch.no_grad():
            for i in range(0, len(sample), 512):
                self.net_(self._images(sample[i:i + 512]).to(self.device_))
        for m in norms:
            m.momentum = 0.1
        self.net_.eval()

    def fit(self, X, y):
        from alphapy.networks import torch_threads
        with torch_threads(self.torch_threads):
            return self._fit(X, y)

    def _fit(self, X, y):
        import torch

        from alphapy.networks import _device
        torch.manual_seed(self.random_state)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError(f"ChartCNNClassifier needs two classes, got {len(self.classes_)}")
        windows, y = self._windows(X), (y == self.classes_[1]).astype(np.float32)
        cut = int(len(y) * (1 - self.validation)) if self.validation else len(y)
        self.device_ = _device(self.device)
        self.net_ = _network(self.window, image_height(self.window)).to(self.device_)
        opt = torch.optim.Adam(self.net_.parameters(), lr=self.lr)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        generator = torch.Generator().manual_seed(self.random_state)
        best, best_state, waited = np.inf, None, 0
        for epoch in range(self.epochs):
            self.net_.train()
            for idx in torch.randperm(cut, generator=generator).split(self.batch_size):
                idx = idx.numpy()
                opt.zero_grad()
                logits = self.net_(self._images(windows[idx]).to(self.device_)).squeeze(-1)
                loss_fn(logits, torch.from_numpy(y[idx]).to(self.device_)).backward()
                opt.step()
            self._recalibrate(windows[:cut])
            if cut == len(y):
                continue
            val = float(loss_fn(self._logits(windows[cut:]), torch.from_numpy(y[cut:])))
            if val < best - 1e-5:
                best, waited = val, 0
                best_state = {k: v.detach().clone() for k, v in self.net_.state_dict().items()}
            else:
                waited += 1
                if waited >= self.patience and epoch + 1 >= self.min_epochs:
                    break
        if best_state is not None:
            self.net_.load_state_dict(best_state)
        self.net_.eval()
        return self

    def predict_proba(self, X):
        from alphapy.networks import torch_threads
        with torch_threads(self.torch_threads):
            return self._predict_proba(X)

    def _predict_proba(self, X):
        import torch
        p = torch.sigmoid(self._logits(self._windows(X))).numpy().astype(np.float64)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]

    def __getstate__(self):
        state = self.__dict__.copy()
        if "net_" in state:
            state["net_weights_"] = {k: v.detach().cpu() for k, v in self.net_.state_dict().items()}
            del state["net_"], state["device_"]
        return state

    def __setstate__(self, state):
        weights = state.pop("net_weights_", None)
        self.__dict__.update(state)
        if weights is not None:
            import torch
            from alphapy.networks import torch_threads
            with torch_threads(1):
                self.net_ = _network(self.window, image_height(self.window))
                self.net_.load_state_dict(weights)
            self.net_.eval()
            self.device_ = torch.device("cpu")
