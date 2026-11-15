"""Raw windows: own group only, nothing after the row, short history masked, sessions respected, scaled within."""

import numpy as np
import pandas as pd
import pytest

from alphapy.sequence import build_windows, flatten, unflatten, window_names, window_shape

SPEC = {"group": "g", "order": "t", "channels": ["close", "volume"], "window": 3,
        "scale": {"close": "close", "volume": "mean"}}


def frame():
    # Two groups interleaved and out of order; group B's prices are far from A's.
    return pd.DataFrame({"g": ["A", "B", "A", "B", "A", "A"], "t": [2, 1, 1, 2, 3, 4],
                         "close": [11.0, 500.0, 10.0, 510.0, 12.0, 13.0],
                         "volume": [200.0, 9.0, 100.0, 11.0, 300.0, 400.0]})


def test_a_window_holds_its_own_groups_bars_up_to_its_own_scaled_within_the_window():
    w = build_windows(frame(), SPEC)
    assert w.shape == (6, 3, 3) and w.dtype == np.float32
    # Row 4 is A at t=3: closes 10, 11, 12 over the last close; volumes 100, 200, 300 over their mean.
    assert np.allclose(w[4, 0], [10 / 12, 11 / 12, 1.0]) and np.allclose(w[4, 1], [0.5, 1.0, 1.5])
    assert np.allclose(w[4, 2], [1, 1, 1])
    # Row 3 is B at t=2: only B's two bars, the oldest slot padded and masked.
    assert np.allclose(w[3, 0], [0.0, 500 / 510, 1.0]) and np.allclose(w[3, 2], [0, 1, 1])


def test_no_window_changes_when_a_later_bar_changes():
    before = build_windows(frame(), SPEC)
    later = frame()
    later.loc[5, ["close", "volume"]] = [99.0, 9999.0]      # A's last bar (t=4)
    after = build_windows(later, SPEC)
    assert np.array_equal(before[:5], after[:5]) and not np.array_equal(before[5], after[5])


def test_a_window_does_not_cross_a_session_change():
    f = frame().assign(s=["d1", "x", "d1", "x", "d2", "d2"])
    w = build_windows(f, {**SPEC, "session": "s"})
    # Row 4 is A's first bar of session d2: its t=1 and t=2 bars belong to d1 and are masked out.
    assert np.allclose(w[4, 2], [0, 0, 1]) and np.allclose(w[4, 0], [0, 0, 1])


def test_the_flat_block_round_trips_and_is_named_channel_by_channel():
    w = build_windows(frame(), SPEC)
    n_ch, win = window_shape(SPEC)
    assert np.array_equal(unflatten(flatten(w), n_ch, win), w)
    assert window_names(SPEC)[:3] == ["close[-2]", "close[-1]", "close[-0]"] and len(window_names(SPEC)) == 9


def test_windows_too_large_for_memory_are_refused():
    with pytest.raises(MemoryError, match="max_gb"):
        build_windows(frame(), {**SPEC, "max_gb": 1e-9})
