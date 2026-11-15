################################################################################
#
# Package   : AlphaPy
# Module    : sequence
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

"""Raw windows for models that learn their own features (the ``sequence`` section of ``model.yml``).

Each row gets the last ``window`` values of each channel from its own group, its own value last, ordered by the
``order`` column; nothing after the row enters its window. Short history is padded and flagged in a mask channel (no
row is dropped). With ``session`` set, values from another session than the row's are treated as missing. Each window
is scaled within itself, so no statistic crosses rows: a channel divided by the last value of a named channel (prices
by the last close) or by its own window mean (volume), or left as is.
"""

#
# Imports
#

import logging
import warnings

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MASK = "_mask"


def window_shape(spec: dict) -> tuple[int, int]:
    """(channels including the mask, window length) for a ``sequence`` specification."""
    return len(spec["channels"]) + 1, int(spec["window"])


def window_names(spec: dict) -> list[str]:
    """Column names of the flattened window block, channel by channel, oldest bar first."""
    channels, window = list(spec["channels"]) + [MASK], int(spec["window"])
    return [f"{c}[-{window - 1 - k}]" for c in channels for k in range(window)]


def build_windows(frame: pd.DataFrame, spec: dict) -> np.ndarray:
    """Windows for every row of ``frame`` in its own order: float32 array (rows, channels + mask, window)."""
    group, order, session = spec["group"], spec["order"], spec.get("session")
    channels, window = list(spec["channels"]), int(spec["window"])
    scale = spec.get("scale") or {}
    n_rows, n_ch = len(frame), len(channels)
    size_gb = n_rows * (n_ch + 1) * window * 4 / 1e9
    limit = float(spec.get("max_gb", 8.0))
    logger.info("Sequence windows: %d rows x %d channels x %d bars = %.2f GB", n_rows, n_ch + 1, window, size_gb)
    if size_gb > limit:
        raise MemoryError(f"Sequence windows need {size_gb:.1f} GB, above the {limit:.1f} GB limit (max_gb)")

    out = np.zeros((n_rows, n_ch + 1, window), dtype=np.float32)
    keys = frame[[group, order]].reset_index(drop=True)
    positions = keys.sort_values([group, order], kind="stable").index.to_numpy()
    values = frame[channels].to_numpy(dtype=np.float64)[positions]
    groups = frame[group].to_numpy()[positions]
    sessions = frame[session].to_numpy()[positions] if session else None
    starts = np.flatnonzero(np.r_[True, groups[1:] != groups[:-1]])
    ends = np.r_[starts[1:], len(positions)]
    for start, end in zip(starts, ends):
        block = values[start:end]
        padded = np.vstack([np.full((window - 1, n_ch), np.nan), block])
        win = np.lib.stride_tricks.sliding_window_view(padded, window, axis=0)   # (bars, channels, window)
        win = win.copy()
        if sessions is not None:
            sess = np.r_[np.full(window - 1, None, dtype=object), sessions[start:end]]
            sess_win = np.lib.stride_tricks.sliding_window_view(sess, window)
            other = sess_win != sess_win[:, -1:]
            win[np.broadcast_to(other[:, None, :], win.shape)] = np.nan
        present = ~np.isnan(win).any(axis=1)                                     # (bars, window)
        win = _scale(win, channels, scale)
        out[positions[start:end], :n_ch, :] = np.nan_to_num(win, nan=0.0, posinf=0.0, neginf=0.0)
        out[positions[start:end], n_ch, :] = present
    return out


def _scale(win: np.ndarray, channels: list[str], scale: dict) -> np.ndarray:
    index = {c: i for i, c in enumerate(channels)}
    scaled = win.copy()
    with np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # an all-missing window has no mean
        for channel, how in scale.items():
            i = index[channel]
            if how in (None, "none"):
                continue
            if how == "mean":
                scaled[:, i, :] = win[:, i, :] / np.nanmean(win[:, i, :], axis=1, keepdims=True)
            else:
                scaled[:, i, :] = win[:, i, :] / win[:, index[how], -1:]
    return scaled


def flatten(windows: np.ndarray) -> np.ndarray:
    """(rows, channels, window) to the 2-D block that joins ``X``."""
    return windows.reshape(windows.shape[0], -1)


def unflatten(block: np.ndarray, n_channels: int, window: int) -> np.ndarray:
    """The 2-D window block back to (rows, channels, window)."""
    return np.asarray(block, dtype=np.float32).reshape(-1, n_channels, window)
