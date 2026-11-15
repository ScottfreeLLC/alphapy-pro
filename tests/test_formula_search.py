"""Formula search: the evaluator's arithmetic, a planted relationship found, reproducible fits on their own rows."""

import random

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("sb3_contrib")
from alphapy.formula_search import FormulaSearchLearner, _Grammar, evaluate, parse, to_string  # noqa: E402

CH = ("close", "volume", "_mask")


def small_windows():
    # Row 1 has two padded bars: their values (9) must never be read.
    close = [[1, 2, 3, 4, 5], [9, 9, 2, 4, 6]]
    volume = [[1, 1, 2, 2, 4], [9, 9, 1, 3, 2]]
    mask = [[1, 1, 1, 1, 1], [0, 0, 1, 1, 1]]
    return np.array([close, volume, mask], dtype=np.float32).transpose(1, 0, 2)


@pytest.mark.parametrize("formula, expected", [
    ("Mean(close, 3)", [4, 4]),
    ("Mean(close, 5)", [3, np.nan]),                       # short history
    ("Ref(close, 2)", [3, 2]),
    ("Delta(close, 1)", [1, 2]),
    ("Max(Ref(close, 1), 3)", [4, np.nan]),                # the shifted series reaches a padded bar
    ("Std(close, 3)", [np.sqrt(2 / 3), np.sqrt(8 / 3)]),
    ("Mad(close, 3)", [2 / 3, 4 / 3]),
    ("Med(volume, 3)", [2, 2]),
    ("WMA(close, 3)", [26 / 6, 28 / 6]),
    ("EMA(close, 2)", [4.75, 5.5]),                       # weights 1/3 and 1 (smoothing 2/3)
    ("Greater(close, 5.5)", [5.5, 6]),
    ("Div(Sub(close, Mean(close, 3)), Std(close, 3))", [1 / np.sqrt(2 / 3), 2 / np.sqrt(8 / 3)]),
    ("Corr(close, volume, 5)", [np.corrcoef([1, 2, 3, 4, 5], [1, 1, 2, 2, 4])[0, 1], np.nan]),
    ("Cov(close, volume, 3)", [np.cov([3, 4, 5], [2, 2, 4], bias=True)[0, 1], np.cov([2, 4, 6], [1, 3, 2], bias=True)[0, 1]]),
])
def test_evaluator_values(formula, expected):
    assert to_string(parse(formula)) == formula
    np.testing.assert_allclose(evaluate(formula, small_windows(), CH), expected, rtol=1e-6)


@pytest.mark.parametrize("formula", ["Div(close, Sub(close, close))", "Log(Neg(close))", "Corr(close, close, 1)",
                                     "Mul(Mul(Mul(volume, 30), Mul(volume, 30)), Div(1, Sub(volume, volume)))"])
def test_invalid_operations_are_missing_then_filled(formula):
    assert np.isnan(evaluate(formula, small_windows(), CH)).all()
    assert (evaluate(formula, small_windows(), CH, fill=0.5) == 0.5).all()


@pytest.mark.parametrize("max_length", [1, 3, 5, 20])
def test_masked_sampling_always_finishes_a_formula(max_length):
    grammar, rng = _Grammar(["close", "volume"], [1, 2, 5], max_length), np.random.default_rng(0)
    for _ in range(300):
        stack, ids = (), []
        while len(ids) < max_length:
            action = rng.choice(np.flatnonzero(grammar.mask(stack, len(ids))))
            if grammar.tokens[action][0] == "sep":
                break
            ids.append(action)
            stack = grammar.push(stack, grammar.tokens[action])
        tree = grammar.tree(ids)
        assert parse(to_string(tree)) == tree
        assert evaluate(tree, small_windows(), CH).shape == (2,)


def planted(n, seed, window=20):
    """Windows scaled as alphapy.sequence builds them (close by its last value, volume by its mean); a tenth of rows
    have short history. The target follows the last close against its 5-bar mean."""
    rng = np.random.default_rng(seed)
    close = np.exp(np.cumsum(0.01 * rng.normal(size=(n, window)), axis=1))
    volume = rng.lognormal(0, 0.5, (n, window))
    mask = np.ones((n, window))
    short = rng.random(n) < 0.1
    mask[short, :int(window * 0.6)] = 0
    signal = close[:, -1] / close[:, -5:].mean(axis=1) - 1
    y = (signal + 0.5 * signal.std() * rng.normal(size=n) > 0).astype(int)
    close, volume = close / close[:, -1:], volume / volume.mean(axis=1, keepdims=True)
    w = np.stack([close * mask, volume * mask, mask], axis=1).astype(np.float32)
    return w, y


def test_planted_relationship_is_found():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    w, y = planted(2000, 0)
    w_new, y_new = planted(2000, 1)
    learner = FormulaSearchLearner(total_timesteps=4096, channels=CH).fit(w, y)   # 4 rollouts: 3 updates feed sampling
    assert learner.ic_ > 0.3 and len(learner.formulas_) == len(learner.weights_) <= 10
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(learner.transform(w), y)
    assert roc_auc_score(y_new, clf.predict_proba(learner.transform(w_new))[:, 1]) > 0.7


def test_fit_uses_only_its_arguments_and_is_reproducible():
    w, y = planted(500, 2)
    w_other, y_other = planted(500, 3)
    copy = w.copy()
    np_state, py_state = np.random.get_state()[1].copy(), random.getstate()
    params = dict(total_timesteps=1024, n_steps=256, channels=CH, random_state=4)   # 3 updates feed sampling
    first = FormulaSearchLearner(**params).fit(w, y)
    reused = FormulaSearchLearner(**params)
    reused.fit(w_other, y_other)
    reused.fit(w, y)
    assert first.formulas_ == reused.formulas_
    np.testing.assert_array_equal(first.weights_, reused.weights_)
    np.testing.assert_array_equal(w, copy)
    assert (np.random.get_state()[1] == np_state).all() and random.getstate() == py_state


def test_each_row_depends_only_on_its_own_window():
    w, y = planted(500, 5)
    learner = FormulaSearchLearner(total_timesteps=1024, channels=CH).fit(w, y)
    out = learner.transform(w)
    assert out.dtype == np.float32 and out.shape == (500, len(learner.formulas_)) and np.isfinite(out).all()
    other = w.copy()
    other[1:] = planted(499, 6)[0]
    np.testing.assert_array_equal(learner.transform(other)[0], out[0])
    np.testing.assert_array_equal(learner.transform(w[:1]), out[:1])
