################################################################################
#
# Package   : AlphaPy
# Module    : kronos_features
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

"""Window features from the frozen Kronos foundation model (Shi et al., "Kronos: A Foundation Model for the Language
of Financial Markets", arXiv 2508.02739), whose model code is vendored in ``alphapy.vendor.kronos`` (optional
dependencies ``torch``, ``einops``, ``huggingface_hub``, ``safetensors``).

Kronos encodes each bar (open, high, low, close, volume, amount) into a pair of discrete tokens with its tokenizer,
then reads the token sequence with a causal transformer trained to predict the next bar's tokens. Nothing is trained
here: ``fit`` only loads the published weights, and a window's features are the transformer's state after its last
bar.

Input. Only the window's present bars (mask channel 1) are fed, oldest first, at most ``max_context`` of the most
recent ones; bars missing in the middle of a window are skipped, not filled. Each row's bars are normalised as
Kronos's own ``KronosPredictor`` does: ``amount = volume * mean(open, high, low, close)`` (volume and amount are zero
without a volume channel), then each of the six columns is standardised over the bars fed (population standard
deviation, plus 1e-5) and clipped to +/-5. AlphaPy's window scaling (prices by the last close, volume by its window
mean) multiplies each channel by one constant per window, which this standardisation removes, so the model sees what
it would see on raw prices up to the 1e-5 term. A channel scaled any other way would not match.

Masking. The vendored attention cannot take a padding mask (it passes one to ``scaled_dot_product_attention``
together with ``is_causal=True``, which torch rejects), but both the tokenizer's encoder and the transformer are
causal. So each row's bars are placed first and the rest of the fixed-length input is filled with zeros; position
``k`` sees only positions up to ``k``, and the state at a row's last bar equals the state from feeding its bars alone.

Features. ``pooling="last"`` takes the final normalised hidden state at the row's last bar, the state Kronos's head
reads to predict the next bar's token (the training objective, and the purpose of these features); ``"mean"`` averages the states
over the row's bars; ``"both"`` concatenates them. The model's calendar embedding is not used (windows carry no
timestamps), although Kronos was trained with one. A row with no present bar gets zeros.
"""

#
# Imports
#

import logging

import numpy as np

logger = logging.getLogger(__name__)

PRICE_COLUMNS = ("open", "high", "low", "close")
# Hugging Face commits the default weights were checked at; other names load their latest commit unless pinned.
REVISIONS = {"NeoQuasar/Kronos-small": "901c26c1332695a2a8f243eb2f37243a37bea320",
             "NeoQuasar/Kronos-Tokenizer-base": "0e0117387f39004a9016484a186a908917e22426"}


class KronosLearner:
    """Feature learner for ``LearnedFeatureClassifier``: frozen Kronos embeddings of each window (fit trains nothing)."""

    def __init__(self, model="NeoQuasar/Kronos-small", tokenizer="NeoQuasar/Kronos-Tokenizer-base",
                 model_revision=None, tokenizer_revision=None,
                 channels=("open", "high", "low", "close", "volume", "_mask"), max_context=512, clip=5.0,
                 pooling="last", batch_size=256, device="auto", torch_threads=1, random_state=0, n_jobs=1):
        self.model, self.tokenizer = model, tokenizer
        self.model_revision, self.tokenizer_revision = model_revision, tokenizer_revision
        self.channels, self.max_context, self.clip, self.pooling = tuple(channels), int(max_context), clip, pooling
        self.batch_size, self.device, self.torch_threads, self.random_state = batch_size, device, torch_threads, random_state
        if pooling not in ("last", "mean", "both"):
            raise ValueError(f"pooling must be 'last', 'mean' or 'both', not {pooling!r}")
        missing = [c for c in PRICE_COLUMNS if c not in self.channels]
        if missing:
            raise ValueError(f"Kronos needs channels {list(PRICE_COLUMNS)}; missing {missing}")

    def fit(self, windows=None, y=None):
        from alphapy.networks import torch_threads
        with torch_threads(self.torch_threads):
            return self._fit(windows, y)

    def _fit(self, windows=None, y=None):

        from alphapy.networks import _device
        from alphapy.vendor.kronos import Kronos, KronosTokenizer
        self.device_ = _device(self.device)
        self.tokenizer_ = KronosTokenizer.from_pretrained(
            self.tokenizer, revision=self.tokenizer_revision or REVISIONS.get(self.tokenizer))
        self.model_ = Kronos.from_pretrained(self.model, revision=self.model_revision or REVISIONS.get(self.model))
        for module in (self.tokenizer_, self.model_):
            module.eval().requires_grad_(False).to(self.device_)
        return self

    def _inputs(self, windows):
        """Kronos's normalised (rows, length, 6) input with each row's bars first, and each row's bar count."""
        n, c, w = windows.shape
        if c != len(self.channels):
            raise ValueError(f"Windows have {c} channels but channels names {len(self.channels)}: {self.channels}")
        index = {name: i for i, name in enumerate(self.channels)}
        present = windows[:, index["_mask"], :] > 0.5 if "_mask" in index else np.ones((n, w), dtype=bool)
        length = min(w, self.max_context)
        counts = present.sum(axis=1)
        # Present bars first in time order, then absent ones; keep the last `length` present bars.
        order = np.argsort(np.where(present, 0, 1) * w + np.arange(w), axis=1, kind="stable")
        start = np.maximum(counts - length, 0)
        take = np.minimum(start[:, None] + np.arange(length), w - 1)
        cols = np.take_along_axis(order, take, axis=1)                                  # (rows, length)
        used = np.minimum(counts, length)
        valid = np.arange(length) < used[:, None]

        def channel(name):
            return np.take_along_axis(windows[:, index[name], :].astype(np.float64), cols, axis=1)

        prices = [channel(name) for name in PRICE_COLUMNS]
        volume = channel("volume") if "volume" in index else np.zeros((n, length))
        amount = volume * np.mean(prices, axis=0) if "volume" in index else np.zeros((n, length))
        x = np.stack([*prices, volume, amount], axis=-1)                               # (rows, length, 6)
        v = valid[:, :, None]
        denom = np.maximum(used, 1)[:, None, None]
        mean = (x * v).sum(axis=1, keepdims=True) / denom
        std = np.sqrt((((x - mean) * v) ** 2).sum(axis=1, keepdims=True) / denom)
        x = np.clip((x - mean) / (std + 1e-5), -self.clip, self.clip)
        return np.where(v, x, 0.0).astype(np.float32), used

    def transform(self, windows):
        from alphapy.networks import torch_threads
        with torch_threads(self.torch_threads):
            return self._transform(windows)

    def _transform(self, windows):
        import torch
        x, used = self._inputs(np.asarray(windows, dtype=np.float32))
        if not len(x):
            return np.zeros((0, self.model_.d_model * (2 if self.pooling == "both" else 1)), dtype=np.float32)
        parts = []
        with torch.no_grad():
            for i in range(0, len(x), self.batch_size):
                batch = torch.from_numpy(x[i:i + self.batch_size]).to(self.device_)
                n_bars = torch.from_numpy(used[i:i + self.batch_size]).to(self.device_)
                s1, s2 = self.tokenizer_.encode(batch, half=True)
                _, states = self.model_.decode_s1(s1, s2, None)                       # (rows, length, d_model)
                rows = torch.arange(len(batch), device=self.device_)
                last = states[rows, (n_bars - 1).clamp(min=0)]
                valid = (torch.arange(states.shape[1], device=self.device_)[None, :] < n_bars[:, None]).to(states.dtype)
                mean = (states * valid[:, :, None]).sum(dim=1) / n_bars.clamp(min=1)[:, None].to(states.dtype)
                out = {"last": last, "mean": mean, "both": torch.cat([last, mean], dim=1)}[self.pooling]
                parts.append((out * (n_bars > 0)[:, None].to(out.dtype)).float().cpu())
        return torch.cat(parts).numpy().astype(np.float32)
