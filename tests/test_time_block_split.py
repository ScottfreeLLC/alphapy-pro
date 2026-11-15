import numpy as np
import pytest

from alphapy.model import TimeBlockSplit


def test_folds_walk_forward_by_block_and_never_train_on_the_test_blocks_or_later():
    # Six dates, three rows each (a pooled panel); two folds; one date of embargo.
    dates = np.repeat(np.array(["d1", "d2", "d3", "d4", "d5", "d6"]), 3)
    folds = list(TimeBlockSplit(dates, n_splits=2, embargo=1).split())
    assert [sorted(set(dates[test])) for _, test in folds] == [["d3", "d4"], ["d5", "d6"]]
    assert [sorted(set(dates[train])) for train, _ in folds] == [["d1"], ["d1", "d2", "d3"]]
    for train, test in folds:
        assert max(dates[train]) < min(dates[test])


def test_rows_out_of_date_order_are_still_split_by_their_dates():
    dates = np.array(["d3", "d1", "d6", "d2", "d5", "d4"])
    train, test = next(TimeBlockSplit(dates, n_splits=2, embargo=0).split())
    assert sorted(dates[train]) == ["d1", "d2"] and sorted(dates[test]) == ["d3", "d4"]


def test_too_few_blocks_is_refused():
    with pytest.raises(ValueError, match="Too few time blocks"):
        list(TimeBlockSplit(np.array(["d1", "d2", "d3"]), n_splits=2, embargo=1).split())
