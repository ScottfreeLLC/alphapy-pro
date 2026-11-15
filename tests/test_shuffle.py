"""Shuffling moves the target with the rows (it used to shuffle the features alone)."""

from types import SimpleNamespace

import numpy as np

from alphapy.data import shuffle_data


def test_shuffling_keeps_each_row_with_its_target():
    X = np.arange(20.0).reshape(10, 2)
    y = np.arange(10) * 2
    model = SimpleNamespace(specs={"seed": 3, "shuffle": True}, X_train=X.copy(), y_train=y.copy())
    shuffle_data(model)
    assert not np.array_equal(model.X_train, X)
    assert np.array_equal(model.X_train[:, 0] / 2 * 2, model.y_train * 1.0)   # row k has X[k,0] == 2k == y[k]
