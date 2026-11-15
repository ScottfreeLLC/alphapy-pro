################################################################################
#
# Package   : AlphaPy
# Module    : networks
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

"""Neural networks that learn their own features from raw windows (optional dependency ``torch``).

``SequenceNetClassifier`` reads the window block as the last ``n_channels * window`` columns of what it is given and
trains one of these networks end to end on the rows ``fit`` receives, holding out the latest ``validation`` share of
them (rows arrive in time order) for early stopping:

- ``cnn``: 1D convolutions with global average pooling.
- ``lstm`` / ``gru``: a recurrent layer read at the last bar.
- ``inception``: InceptionTime blocks (Ismail Fawaz et al., 2020): parallel convolutions of several widths.
- ``tcn``: dilated causal convolutions with residual connections (Bai, Kolter and Koltun, 2018).
- ``patch``: a transformer encoder over patches of bars (after PatchTST, Nie et al., 2023).
- ``cnn_lstm``: convolutions, then an LSTM over their outputs (a bar-based design in the spirit of DeepLOB's
  convolution-then-LSTM layout; DeepLOB itself reads order books).
"""

#
# Imports
#

import logging

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from alphapy.sequence import unflatten

logger = logging.getLogger(__name__)


def _device(name):
    import torch
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)


def _network(arch, n_channels, window, hidden):
    import torch
    from torch import nn

    class GlobalPool(nn.Module):
        def forward(self, x):
            return x.mean(dim=-1)

    class LastStep(nn.Module):
        def forward(self, x):
            return x[..., -1]

    class Last(nn.Module):
        def __init__(self, rnn):
            super().__init__()
            self.rnn = rnn

        def forward(self, x):                      # x: (batch, channels, time)
            out, _ = self.rnn(x.transpose(1, 2))
            return out[:, -1, :]

    class Inception(nn.Module):
        def __init__(self, c_in, c_out):
            super().__init__()
            self.bottleneck = nn.Conv1d(c_in, c_out, 1, bias=False)
            self.convs = nn.ModuleList(nn.Conv1d(c_out, c_out, k, padding=k // 2, bias=False) for k in (9, 19, 39))
            self.pool = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1), nn.Conv1d(c_in, c_out, 1, bias=False))
            self.norm = nn.Sequential(nn.BatchNorm1d(4 * c_out), nn.ReLU())

        def forward(self, x):
            b = self.bottleneck(x)
            width = x.shape[-1]
            parts = [conv(b)[..., :width] for conv in self.convs] + [self.pool(x)]
            return self.norm(torch.cat(parts, dim=1))

    class Causal(nn.Module):
        def __init__(self, c_in, c_out, dilation):
            super().__init__()
            self.pad = 2 * dilation
            self.conv = nn.Conv1d(c_in, c_out, 3, dilation=dilation)
            self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

        def forward(self, x):
            return torch.relu(self.conv(nn.functional.pad(x, (self.pad, 0))) + self.skip(x))

    class Patches(nn.Module):
        def __init__(self, c_in, d, window, patch=4):
            super().__init__()
            self.patch = min(patch, window)
            self.embed = nn.Linear(c_in * self.patch, d)
            n_patches = window // self.patch
            self.position = nn.Parameter(torch.zeros(1, n_patches, d))
            layer = nn.TransformerEncoderLayer(d, nhead=4, dim_feedforward=2 * d, batch_first=True, dropout=0.1)
            self.encoder = nn.TransformerEncoder(layer, num_layers=2)

        def forward(self, x):                      # use the latest whole patches
            b, c, t = x.shape
            n = t // self.patch
            x = x[..., t - n * self.patch:].reshape(b, c, n, self.patch).permute(0, 2, 1, 3).reshape(b, n, -1)
            return self.encoder(self.embed(x) + self.position).mean(dim=1)

    h = hidden
    if arch == "cnn":
        body = nn.Sequential(nn.Conv1d(n_channels, h, 5, padding=2), nn.ReLU(), nn.Conv1d(h, h, 5, padding=2),
                             nn.ReLU(), nn.Conv1d(h, h, 3, padding=1), nn.ReLU(), GlobalPool())
    elif arch in ("lstm", "gru"):
        rnn = (nn.LSTM if arch == "lstm" else nn.GRU)(n_channels, h, num_layers=2, batch_first=True, dropout=0.1)
        body = Last(rnn)
    elif arch == "inception":
        body = nn.Sequential(Inception(n_channels, h // 4), Inception(h, h // 4), Inception(h, h // 4), GlobalPool())
    elif arch == "tcn":
        body = nn.Sequential(Causal(n_channels, h, 1), Causal(h, h, 2), Causal(h, h, 4), Causal(h, h, 8), LastStep())
    elif arch == "patch":
        body = Patches(n_channels, h, window)
    elif arch == "cnn_lstm":
        body = nn.Sequential(nn.Conv1d(n_channels, h, 3, padding=1), nn.ReLU(), nn.Conv1d(h, h, 3, padding=1),
                             nn.ReLU(), Last(nn.LSTM(h, h, batch_first=True)))
    else:
        raise ValueError(f"Unknown architecture {arch}")
    return nn.Sequential(body, nn.Linear(h, 1))


class SequenceNetClassifier(ClassifierMixin, BaseEstimator):
    """A network trained end to end on the window block of the rows ``fit`` receives."""

    def __init__(self, arch="cnn", n_channels=6, window=20, hidden=64, epochs=30, batch_size=512, lr=1e-3,
                 weight_decay=1e-4, validation=0.1, patience=5, min_epochs=10, device="auto", torch_threads=1,
                 random_state=0):
        self.arch = arch
        self.n_channels = n_channels
        self.window = window
        self.hidden = hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.validation = validation
        self.patience = patience
        self.min_epochs = min_epochs
        self.device = device
        self.torch_threads = torch_threads
        self.random_state = random_state

    def _windows(self, X):
        width = self.n_channels * self.window
        windows = unflatten(np.asarray(X, dtype=np.float32)[:, -width:], self.n_channels, self.window)
        if hasattr(self, "mean_"):                 # each channel standardised by the fitted rows' statistics
            windows = (windows - self.mean_) / self.std_
        return windows

    def fit(self, X, y):
        import torch

        # LightGBM and torch each bundle an OpenMP runtime; in one process torch's CPU kernels can deadlock unless
        # torch keeps to one thread. Training runs on MPS when available, so this costs little.
        torch.set_num_threads(self.torch_threads)
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        for name in ("mean_", "std_"):
            self.__dict__.pop(name, None)
        raw = self._windows(X)
        self.mean_ = raw.mean(axis=(0, 2), keepdims=True)[0]
        self.std_ = raw.std(axis=(0, 2), keepdims=True)[0] + 1e-6
        windows, y = self._windows(X), np.asarray(y).ravel().astype(np.float32)
        self.classes_ = np.array([0, 1])
        cut = int(len(y) * (1 - self.validation)) if self.validation else len(y)
        device = _device(self.device)
        net = _network(self.arch, self.n_channels, self.window, self.hidden).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = torch.nn.BCEWithLogitsLoss()
        train_x, train_y = torch.from_numpy(windows[:cut]), torch.from_numpy(y[:cut])
        val_x, val_y = torch.from_numpy(windows[cut:]), torch.from_numpy(y[cut:])
        best, best_state, waited = np.inf, None, 0
        generator = torch.Generator().manual_seed(self.random_state)
        for epoch in range(self.epochs):
            net.train()
            for idx in torch.randperm(cut, generator=generator).split(self.batch_size):
                opt.zero_grad()
                loss = loss_fn(net(train_x[idx].to(device)).squeeze(-1), train_y[idx].to(device))
                loss.backward()
                opt.step()
            if len(val_y) == 0:
                continue
            val = float(loss_fn(self._logits(net, val_x, device), val_y.to(device)))
            if val < best - 1e-5:
                best, waited = val, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                waited += 1
                # Recurrent networks sit on a flat loss for their first epochs; stopping there keeps untrained weights.
                if waited >= self.patience and epoch + 1 >= self.min_epochs:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        self.net_, self.device_ = net.eval(), device
        return self

    def __getstate__(self):
        # A fitted model pickles with the run as its weights on the CPU (the layers are built inside a function and
        # MPS tensors do not pickle); loading rebuilds the network, which then predicts on the CPU.
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
            self.net_ = _network(self.arch, self.n_channels, self.window, self.hidden)
            self.net_.load_state_dict(weights)
            self.net_.eval()
            self.device_ = torch.device("cpu")

    def _logits(self, net, x, device):
        import torch

        net.eval()
        with torch.no_grad():
            return torch.cat([net(chunk.to(device)).squeeze(-1) for chunk in x.split(4096)])

    def predict_proba(self, X):
        import torch

        torch.set_num_threads(self.torch_threads)
        logits = self._logits(self.net_, torch.from_numpy(self._windows(X)), self.device_)
        p = torch.sigmoid(logits).cpu().numpy().astype(np.float64)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
