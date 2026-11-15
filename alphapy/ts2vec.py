################################################################################
#
# Package   : AlphaPy
# Module    : ts2vec
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

"""Self-supervised window representations after TS2Vec (Yue et al., "TS2Vec: Towards Universal Representation of
Time Series", AAAI 2022), reimplemented from the paper (optional dependency ``torch``).

An encoder (a linear input projection, random timestamp masking during training, then residual dilated convolutions)
maps a window to a representation per bar. Training needs no labels: two overlapping random crops of each window are
encoded, and a hierarchical contrastive loss pulls the two views of the same bar together and pushes apart other bars
(temporal contrast) and other windows (instance contrast), repeated after each halving of the time axis by max
pooling. A window's features are its representations max-pooled over time.
"""

#
# Imports
#

import logging

import numpy as np

logger = logging.getLogger(__name__)


def _encoder(n_channels, hidden, out, depth):
    import torch
    from torch import nn

    class Block(nn.Module):
        def __init__(self, c_in, c_out, dilation):
            super().__init__()
            self.conv1 = nn.Conv1d(c_in, c_out, 3, padding=dilation, dilation=dilation)
            self.conv2 = nn.Conv1d(c_out, c_out, 3, padding=dilation, dilation=dilation)
            self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

        def forward(self, x):
            h = self.conv1(nn.functional.gelu(x))
            return self.conv2(nn.functional.gelu(h)) + self.skip(x)

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.project = nn.Linear(n_channels, hidden)
            widths = [hidden] * depth + [out]
            self.blocks = nn.Sequential(*[Block(widths[i - 1] if i else hidden, widths[i], 2 ** i)
                                          for i in range(len(widths))])

        def forward(self, x, mask=False):          # x: (batch, time, channels) -> (batch, time, out)
            h = self.project(x)
            if mask:                               # the paper's binomial timestamp masking (p = 0.5)
                keep = torch.bernoulli(torch.full(h.shape[:2], 0.5, device=h.device)).bool()
                h = h * keep.unsqueeze(-1)
            return self.blocks(h.transpose(1, 2)).transpose(1, 2)

    return Encoder()


def _instance_loss(z1, z2):
    import torch
    import torch.nn.functional as F

    b = z1.size(0)
    if b == 1:
        return z1.new_tensor(0.0)
    z = torch.cat([z1, z2], dim=0).transpose(0, 1)             # (time, 2b, dim)
    sim = torch.matmul(z, z.transpose(1, 2))
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1] + torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    i = torch.arange(b, device=z1.device)
    return (logits[:, i, b + i - 1].mean() + logits[:, b + i, i].mean()) / 2


def _temporal_loss(z1, z2):
    import torch
    import torch.nn.functional as F

    t = z1.size(1)
    if t == 1:
        return z1.new_tensor(0.0)
    z = torch.cat([z1, z2], dim=1)                             # (batch, 2t, dim)
    sim = torch.matmul(z, z.transpose(1, 2))
    logits = torch.tril(sim, diagonal=-1)[:, :, :-1] + torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)
    i = torch.arange(t, device=z1.device)
    return (logits[:, i, t + i - 1].mean() + logits[:, t + i, i].mean()) / 2


def hierarchical_loss(z1, z2):
    """The paper's loss: instance and temporal contrast, repeated after each halving of the time axis."""
    import torch.nn.functional as F

    loss, levels = 0.0, 0
    while z1.size(1) > 1:
        loss = loss + _instance_loss(z1, z2) + _temporal_loss(z1, z2)
        levels += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), 2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), 2).transpose(1, 2)
    if z1.size(1) == 1:
        loss = loss + _instance_loss(z1, z2)
        levels += 1
    return loss / levels


class TS2VecLearner:
    """Feature learner for ``LearnedFeatureClassifier``: fit trains the encoder on the given windows (no labels)."""

    def __init__(self, hidden=64, out=128, depth=4, epochs=20, batch_size=256, lr=1e-3, device="auto",
                 torch_threads=1, random_state=0, n_jobs=1):
        self.hidden, self.out, self.depth = hidden, out, depth
        self.epochs, self.batch_size, self.lr = epochs, batch_size, lr
        self.device, self.torch_threads, self.random_state = device, torch_threads, random_state

    def fit(self, windows, y=None):
        from alphapy.networks import torch_threads
        with torch_threads(self.torch_threads):
            return self._fit(windows, y)

    def _fit(self, windows, y=None):
        import torch

        from alphapy.networks import _device
        torch.manual_seed(self.random_state)
        rng = np.random.default_rng(self.random_state)
        self.device_ = _device(self.device)
        x = torch.from_numpy(np.ascontiguousarray(windows.transpose(0, 2, 1)))   # (rows, time, channels)
        self.encoder_ = _encoder(x.shape[2], self.hidden, self.out, self.depth).to(self.device_)
        opt = torch.optim.AdamW(self.encoder_.parameters(), lr=self.lr)
        t = x.shape[1]
        for _ in range(self.epochs):
            self.encoder_.train()
            for idx in torch.randperm(len(x), generator=torch.Generator().manual_seed(int(rng.integers(1 << 30)))
                                      ).split(self.batch_size):
                batch = x[idx].to(self.device_)
                # The paper's two views: [a1, b1) and [a2, b2) overlap on [a2, b1), the last `crop` bars of the first
                # and the first `crop` bars of the second (a1 <= a2 < b1 <= b2).
                crop = int(rng.integers(2, t + 1))
                a2 = int(rng.integers(0, t - crop + 1))
                b1 = a2 + crop
                a1 = int(rng.integers(0, a2 + 1))
                b2 = int(rng.integers(b1, t + 1))
                z1 = self.encoder_(batch[:, a1:b1], mask=True)[:, -crop:]
                z2 = self.encoder_(batch[:, a2:b2], mask=True)[:, :crop]
                opt.zero_grad()
                hierarchical_loss(z1, z2).backward()
                opt.step()
        self.encoder_.eval()
        return self

    def transform(self, windows):
        from alphapy.networks import torch_threads
        with torch_threads(self.torch_threads):
            return self._transform(windows)

    def _transform(self, windows):
        import torch
        x = torch.from_numpy(np.ascontiguousarray(windows.transpose(0, 2, 1)))
        with torch.no_grad():
            parts = [self.encoder_(x[i:i + 4096].to(self.device_)).max(dim=1).values.cpu()
                     for i in range(0, len(x), 4096)]
        return torch.cat(parts).numpy().astype(np.float32)
