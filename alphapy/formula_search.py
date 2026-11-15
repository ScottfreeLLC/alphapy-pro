################################################################################
#
# Package   : AlphaPy
# Module    : formula_search
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

"""Formula features found by reinforcement learning, after Yu et al., "Generating Synergistic Formulaic Alpha
Collections via Reinforcement Learning" (KDD 2023, arXiv:2306.12964). The code is a reimplementation from the paper
(optional dependencies ``torch``, ``gymnasium``, ``stable-baselines3`` and ``sb3-contrib``).

A formula is a token sequence in reverse Polish notation over the window's channels, constants, time spans and the
paper's operators (Table 4) plus ``Sign`` and ``Neg``. It is evaluated along the window's time axis, one series per
row, and its value is the series at the row's own (last) bar, so it reads only that row's window. A bar flagged
missing by the mask channel is missing in every series; a rolling operator needs all of its bars, a division by
(nearly) zero, the log of a non-positive value or a magnitude of 1e10 or more is missing, and a row whose value is missing takes the formula's
mean over the fit rows. New rows are clipped to the range of the formula's values on the fit rows.

A policy (an LSTM over the tokens so far, then policy and value heads) is trained with PPO and invalid-action
masking (sb3-contrib's ``MaskablePPO``) to write formulas; the mask admits only tokens that keep the sequence a
well-formed expression completable within ``max_formula_length`` tokens. A finished formula is offered to a pool of
at most ``pool_size`` formulas combined linearly (standardised values, least-squares weights fitted to the
standardised target). The episode's reward is the gain in the combination's information coefficient (its Pearson
correlation with the target on the fit rows) from offering the formula; when the pool is full the formula with the
smallest absolute weight leaves it. Differences from the paper:

- Reward: the gain in combined IC. The paper's Algorithm 2 returns the combined IC after the update; with a pool that
  persists between episodes that carries a slowly moving baseline, which the gain removes.
- IC: Pearson correlation over all fit rows. The paper averages daily cross-sectional ICs; a learner sees windows
  and targets only, with no dates.
- Weights: exact least squares (normal equations with a tiny ridge) rather than gradient descent from a random
  start, and refitted after a formula leaves the pool rather than kept.
- A formula whose fit-row values are constant (for example ``Sub(close, close)``) is invalid and earns the paper's
  -1 for semantically invalid expressions; one equivalent to a pool formula by its values earns 0 and is not added.
- Time spans are chosen from the window length (the paper's 10 to 50 days do not fit short windows).
"""

#
# Imports
#

import logging
import math
import random
import re

import numpy as np

logger = logging.getLogger(__name__)

MASK = "_mask"
CHANNELS = ("open", "high", "low", "close", "volume", MASK)
CONSTANTS = (-30.0, -10.0, -5.0, -2.0, -1.0, -0.5, -0.01, 0.01, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)
TIME_SPANS = (1, 2, 3, 5, 10, 20, 30, 40, 50)

UNARY = ("Abs", "Log", "Sign", "Neg")
BINARY = ("Add", "Sub", "Mul", "Div", "Greater", "Less")
ROLLING = ("Ref", "Mean", "Sum", "Std", "Var", "Max", "Min", "Med", "Mad", "Delta", "WMA", "EMA")
ROLLING_PAIR = ("Cov", "Corr")
ARITY = {**{o: 1 for o in UNARY}, **{o: 2 for o in BINARY}, **{o: 2 for o in ROLLING},
         **{o: 3 for o in ROLLING_PAIR}}

INVALID_REWARD = -1.0      # the paper's reward for an expression that cannot be evaluated
DUPLICATE_CORR = 0.9999    # standardised values this correlated with a pool formula are the same formula
_TINY = 1e-8
_HUGE = 1e10


#
# Expressions: a tree is a channel name (str), a constant (float), or (operator, *operands), the last operand of a
# rolling operator being its time span (int).
#

def to_string(tree) -> str:
    if isinstance(tree, str):
        return tree
    if isinstance(tree, (int, float)):
        return f"{tree:g}"
    return f"{tree[0]}({', '.join(to_string(a) for a in tree[1:])})"


def parse(text: str):
    """The tree of a formula written as ``to_string`` writes it, e.g. ``"Div(Mean(close, 5), Ref(close, 10))"``."""
    tokens = re.findall(r"[A-Za-z_][A-Za-z_0-9]*|-?\d+(?:\.\d*)?(?:e-?\d+)?|[(),]", text)
    pos = 0

    def node():
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        if pos < len(tokens) and tokens[pos] == "(":
            if tok not in ARITY:
                raise ValueError(f"Unknown operator {tok}")
            pos += 1
            args = [node()]
            while tokens[pos] == ",":
                pos += 1
                args.append(node())
            pos += 1                                     # ")"
            if len(args) != ARITY[tok]:
                raise ValueError(f"{tok} takes {ARITY[tok]} operands")
            if tok in ROLLING or tok in ROLLING_PAIR:
                args[-1] = int(args[-1])
            return (tok, *args)
        if re.fullmatch(r"-?\d.*", tok):
            return float(tok)
        return tok

    tree = node()
    if pos != len(tokens):
        raise ValueError(f"Trailing tokens in {text!r}")
    return tree


def _finite(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(np.isfinite(x) & (np.abs(x) < _HUGE), x, np.nan)


def _shift(x, d):
    out = np.full_like(x, np.nan)
    if d < x.shape[1]:
        out[:, d:] = x[:, :x.shape[1] - d]
    return out


def _rolling(x, d, fn):
    """``fn`` over each run of the last ``d`` bars (axis -1 of the strided view, oldest first), missing before."""
    out = np.full_like(x, np.nan)
    if d <= x.shape[1]:
        out[:, d - 1:] = fn(np.lib.stride_tricks.sliding_window_view(x, d, axis=1))
    return out


def _weighted(x, d, weights):
    w = np.asarray(weights, dtype=np.float64)
    return _rolling(x, d, lambda v: v @ (w / w.sum()))


def _apply(op, args):
    if op == "Abs":
        return np.abs(args[0])
    if op == "Log":
        return np.log(np.where(args[0] > 0, args[0], np.nan))
    if op == "Sign":
        return np.sign(args[0])
    if op == "Neg":
        return -args[0]
    if op in BINARY:
        a, b = args
        if op == "Div":
            return a / np.where(np.abs(b) > _TINY, b, np.nan)
        return {"Add": np.add, "Sub": np.subtract, "Mul": np.multiply,
                "Greater": np.maximum, "Less": np.minimum}[op](a, b)
    x, d = args[0], args[-1]
    if op == "Ref":
        return _shift(x, d)
    if op == "Delta":
        return x - _shift(x, d)
    if op == "Mean":
        return _rolling(x, d, lambda v: v.mean(axis=-1))
    if op == "Sum":
        return _rolling(x, d, lambda v: v.sum(axis=-1))
    if op == "Std":
        return _rolling(x, d, lambda v: v.std(axis=-1))
    if op == "Var":
        return _rolling(x, d, lambda v: v.var(axis=-1))
    if op == "Max":
        return _rolling(x, d, lambda v: v.max(axis=-1))
    if op == "Min":
        return _rolling(x, d, lambda v: v.min(axis=-1))
    if op == "Med":
        return _rolling(x, d, lambda v: np.median(v, axis=-1))
    if op == "Mad":
        return _rolling(x, d, lambda v: np.abs(v - v.mean(axis=-1, keepdims=True)).mean(axis=-1))
    if op == "WMA":                                       # linear weights, the latest bar heaviest
        return _weighted(x, d, np.arange(1, d + 1))
    if op == "EMA":                                       # smoothing 2 / (d + 1), truncated to d bars
        alpha = 2.0 / (d + 1)
        return _weighted(x, d, (1 - alpha) ** np.arange(d - 1, -1, -1))
    y = args[1]
    if op == "Cov":
        return _rolling(x * y, d, lambda v: v.mean(axis=-1)) - _rolling(x, d, lambda v: v.mean(axis=-1)) * \
            _rolling(y, d, lambda v: v.mean(axis=-1))

    def corr(v):                                          # v: (rows, bars, 2, d)
        c = v - v.mean(axis=-1, keepdims=True)
        den = np.sqrt((c[..., 0, :] ** 2).sum(-1) * (c[..., 1, :] ** 2).sum(-1))
        return (c[..., 0, :] * c[..., 1, :]).sum(-1) / np.where(den > _TINY, den, np.nan)

    out = np.full_like(x, np.nan)
    if d <= x.shape[1]:
        out[:, d - 1:] = corr(np.lib.stride_tricks.sliding_window_view(np.stack([x, y], axis=2), d, axis=1))
    return out


def evaluate(formula, windows, channels=CHANNELS, fill=None) -> np.ndarray:
    """Each row's value of ``formula`` (a tree or its string) at its window's last bar: float64 (rows,).

    Missing values (short history, guarded operations) are NaN, or ``fill`` when given.
    """
    tree = parse(formula) if isinstance(formula, str) else formula
    windows = np.asarray(windows, dtype=np.float64)
    names = list(channels)
    if windows.ndim != 3 or windows.shape[1] != len(names):
        raise ValueError(f"Windows of shape {windows.shape} for channels {names}")
    present = windows[:, names.index(MASK), :] > 0.5 if MASK in names else np.ones(windows.shape[::2], dtype=bool)
    width = windows.shape[2]

    def walk(node):
        if isinstance(node, str):
            return np.where(present, windows[:, names.index(node), :], np.nan)
        if isinstance(node, (int, float)):
            return np.full((len(windows), width), float(node))
        op, *operands = node
        args = [walk(a) for a in (operands[:-1] if op in ROLLING or op in ROLLING_PAIR else operands)]
        if op in ROLLING or op in ROLLING_PAIR:
            args.append(int(operands[-1]))
        return _finite(_apply(op, args))

    try:
        with np.errstate(all="ignore"):
            value = walk(tree)[:, -1]
    finally:
        walk = None     # the recursive closure refers to itself; break the cycle so the float64 windows free at once
    return value if fill is None else np.where(np.isnan(value), fill, value)


#
# Tokens and the grammar mask
#

class _Grammar:
    """Token vocabulary and the rules that keep a reverse Polish sequence a completable expression.

    Stack items are "x" (an expression that depends on the data), "c" (a constant) or "t" (a time span). A time span
    may only follow an expression and must be consumed at once by a rolling operator; operators never take only
    constants (that would be a constant expression) and a constant never follows a constant, so a stack of k items
    with an expression always finishes in k - 1 binary operators.
    """

    def __init__(self, features, spans, max_length):
        self.max_length = max_length
        self.tokens = ([("sep", None)] + [("feature", f) for f in features] + [("const", c) for c in CONSTANTS]
                       + [("span", d) for d in spans] + [("op", o) for o in UNARY + BINARY + ROLLING + ROLLING_PAIR])
        self.n = len(self.tokens)

    @staticmethod
    def push(stack, token):
        kind, value = token
        if kind == "feature":
            return stack + ("x",)
        if kind == "const":
            return stack + ("c",)
        if kind == "span":
            return stack + ("t",)
        k = ARITY[value]
        return stack[:-k] + ("x",)

    @staticmethod
    def legal(stack, token):
        kind, value = token
        top = stack[-1] if stack else None
        if kind == "sep":
            return stack == ("x",)
        if top == "t":                       # a time span must be consumed by a rolling operator now
            if kind != "op":
                return False
            if value in ROLLING:
                return len(stack) >= 2 and stack[-2] == "x"
            return value in ROLLING_PAIR and len(stack) >= 3 and stack[-2] == stack[-3] == "x"
        if kind == "feature":
            return True
        if kind == "const":
            return top != "c"
        if kind == "span":
            return top == "x"
        if value in UNARY:
            return top == "x"
        if value in BINARY:
            return len(stack) >= 2 and "x" in stack[-2:]
        return False                         # rolling operators need a time span on top

    @staticmethod
    def need(stack):
        """Fewest tokens that finish ``stack`` as one expression."""
        if stack and stack[-1] == "t":
            return 1 + _Grammar.need(stack[:-2] + ("x",))
        if not stack:
            return 1
        return 2 if stack == ("c",) else len(stack) - 1

    def mask(self, stack, length):
        room = self.max_length - length - 1
        out = np.zeros(self.n, dtype=bool)
        for i, token in enumerate(self.tokens):
            if not self.legal(stack, token):
                continue
            out[i] = token[0] == "sep" or self.need(self.push(stack, token)) <= room
        return out

    def tree(self, ids):
        stack = []
        for i in ids:
            kind, value = self.tokens[i]
            if kind == "sep":
                break
            if kind == "op":
                k = ARITY[value]
                stack, args = stack[:-k], stack[-k:]
                stack.append((value, *args))
            else:
                stack.append(value)
        if len(stack) != 1:
            raise ValueError("Not a complete expression")
        return stack[0]


#
# The pool of formulas and its linear combination
#

class _Pool:
    """At most ``size`` formulas, standardised on the fit rows, combined by least squares on the standardised target."""

    def __init__(self, windows, y, channels, size):
        self.windows, self.channels, self.size = windows, channels, size
        self.y = (y - y.mean()) / y.std()
        self.n = len(y)
        self.formulas, self.fills = [], []              # fills: (mean, lowest, highest) on the fit rows
        self.values = np.empty((self.n, 0))            # standardised values, one column per formula
        self.weights = np.empty(0)
        self.ic = 0.0

    def _fit(self, values):
        corr = values.T @ values / self.n
        ics = values.T @ self.y / self.n
        w = np.linalg.solve(corr + 1e-6 * np.eye(len(ics)), ics)
        spread = math.sqrt(max(float(w @ corr @ w), _TINY))
        return w, float(w @ ics) / spread

    def offer(self, tree):
        """Offer a formula; return the reward (the gain in combined IC, or the invalid-formula reward)."""
        with np.errstate(all="ignore"):      # Accelerate's matmul raises spurious floating-point warnings
            return self._offer(tree)

    def _offer(self, tree):
        text = to_string(tree)
        if text in self.formulas:
            return 0.0
        raw = evaluate(tree, self.windows, self.channels)
        finite = ~np.isnan(raw)
        if not finite.any():
            return INVALID_REWARD
        fill = (float(raw[finite].mean()), float(raw[finite].min()), float(raw[finite].max()))
        raw = np.where(finite, raw, fill[0])
        sd = raw.std()
        if not sd > 1e-5 * max(1.0, abs(fill[0])):    # transform returns float32
            return INVALID_REWARD
        z = (raw - raw.mean()) / sd
        if self.formulas and np.max(np.abs(self.values.T @ z / self.n)) >= DUPLICATE_CORR:
            return 0.0
        formulas, fills = self.formulas + [text], self.fills + [fill]
        values = np.column_stack([self.values, z])
        w, ic = self._fit(values)
        if len(formulas) > self.size:
            drop = int(np.argmin(np.abs(w)))
            keep = [i for i in range(len(formulas)) if i != drop]
            formulas, fills, values = [formulas[i] for i in keep], [fills[i] for i in keep], values[:, keep]
            w, ic = self._fit(values)
        gain = ic - self.ic
        self.formulas, self.fills, self.values, self.weights, self.ic = formulas, fills, values, w, ic
        return gain


#
# The environment and the policy's feature extractor (built lazily: gymnasium and torch are optional)
#

def _environment(grammar, pool):
    import gymnasium as gym

    class FormulaEnv(gym.Env):
        """One episode writes one formula; the reward comes at its end."""

        def __init__(self):
            self.action_space = gym.spaces.Discrete(grammar.n)
            # token ids + 2, after a begin marker (1); 0 pads
            self.observation_space = gym.spaces.Box(0, grammar.n + 1, shape=(grammar.max_length + 1,),
                                                    dtype=np.int64)
            self.invalid = self.episodes = 0

        def _obs(self):
            obs = np.zeros(grammar.max_length + 1, dtype=np.int64)
            obs[0] = 1
            obs[1:len(self.ids) + 1] = np.asarray(self.ids, dtype=np.int64) + 2
            return obs

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.ids, self.stack = [], ()
            return self._obs(), {}

        def action_masks(self):
            return grammar.mask(self.stack, len(self.ids))

        def step(self, action):
            token = grammar.tokens[int(action)]
            if token[0] != "sep":
                self.ids.append(int(action))
                self.stack = grammar.push(self.stack, token)
                if len(self.ids) < grammar.max_length:
                    return self._obs(), 0.0, False, False, {}
            reward = pool.offer(grammar.tree(self.ids))
            self.episodes += 1
            self.invalid += reward == INVALID_REWARD
            return self._obs(), float(reward), True, False, {}

    return FormulaEnv()


def _extractor_class():
    import torch
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from torch import nn

    class TokenLSTM(BaseFeaturesExtractor):
        """The paper's shared feature extractor: a two-layer LSTM (dropout 0.1) over the tokens so far."""

        def __init__(self, observation_space, n_tokens, hidden=128):
            super().__init__(observation_space, features_dim=hidden)
            self.embed = nn.Embedding(n_tokens + 2, hidden, padding_idx=0)
            self.lstm = nn.LSTM(hidden, hidden, num_layers=2, dropout=0.1, batch_first=True)

        def forward(self, obs):
            ids = obs.long()
            out, _ = self.lstm(self.embed(ids))
            last = (ids > 0).sum(dim=1) - 1                 # the latest token (the begin marker when empty)
            return out[torch.arange(len(ids)), last]

    return TokenLSTM


#
# The learner
#

class FormulaSearchLearner:
    """Feature learner for ``LearnedFeatureClassifier``: fit searches formulas on the given rows; transform evaluates
    the pool's formulas, one column each.

    After fit, ``formulas_`` holds the pool as strings (readable and accepted by ``evaluate``), ``weights_`` the
    combination's weights on the standardised formulas, ``fills_`` each formula's value for rows where it is missing,
    ``ranges_`` its (lowest, highest) value on the fit rows, to which transform clips new rows,
    ``ic_`` the combination's information coefficient on the fit rows and ``invalid_share_`` the share of episodes,
    first and second half of training, that wrote an invalid formula.
    """

    def __init__(self, pool_size=10, total_timesteps=20000, max_formula_length=20, channels=CHANNELS,
                 time_spans=None, n_steps=1024, batch_size=128, n_epochs=10, learning_rate=3e-4, ent_coef=0.01,
                 hidden=128, device="cpu", torch_threads=1, random_state=0, n_jobs=1):
        self.pool_size, self.total_timesteps, self.max_formula_length = pool_size, total_timesteps, max_formula_length
        self.channels, self.time_spans = tuple(channels), time_spans
        self.n_steps, self.batch_size, self.n_epochs = n_steps, batch_size, n_epochs
        self.learning_rate, self.ent_coef, self.hidden = learning_rate, ent_coef, hidden
        self.device, self.torch_threads, self.random_state = device, torch_threads, random_state

    def fit(self, windows, y):
        import torch
        from sb3_contrib import MaskablePPO

        windows = np.asarray(windows, dtype=np.float32)
        y = np.asarray(y, dtype=np.float64).ravel()
        if windows.ndim != 3 or windows.shape[1] != len(self.channels):
            raise ValueError(f"Windows of shape {windows.shape} for channels {list(self.channels)}")
        if len(y) != len(windows) or not y.std() > 0:
            raise ValueError("The target needs one varying value per window")
        width = windows.shape[2]
        spans = self.time_spans or [d for d in TIME_SPANS if d <= width]
        features = [c for c in self.channels if c != MASK]
        grammar = _Grammar(features, spans, self.max_formula_length)
        pool = _Pool(windows, y, self.channels, self.pool_size)
        env = _environment(grammar, pool)

        # Whole rollouts that add up to the budget (learn() runs whole rollouts).
        rollouts = max(1, round(self.total_timesteps / self.n_steps))
        n_steps = max(2, self.total_timesteps // rollouts)
        saved = random.getstate(), np.random.get_state(), torch.get_rng_state()   # sb3 seeds the global generators
        threads = torch.get_num_threads()
        torch.set_num_threads(self.torch_threads)    # LightGBM's and torch's OpenMP (see alphapy.networks)
        halves = []
        try:
            model = MaskablePPO(
                "MlpPolicy", env, n_steps=n_steps, batch_size=min(self.batch_size, n_steps), n_epochs=self.n_epochs,
                learning_rate=self.learning_rate, ent_coef=self.ent_coef, gamma=1.0, clip_range=0.2,
                seed=self.random_state, device=self.device, verbose=0,
                policy_kwargs=dict(features_extractor_class=_extractor_class(),
                                   features_extractor_kwargs=dict(n_tokens=grammar.n, hidden=self.hidden),
                                   net_arch=dict(pi=[64, 64], vf=[64, 64])))
            for part in (rollouts // 2, rollouts - rollouts // 2):
                if part:
                    before = env.invalid, env.episodes
                    model.learn(part * n_steps, reset_num_timesteps=False)
                    halves.append((env.invalid - before[0]) / max(1, env.episodes - before[1]))
        finally:
            torch.set_num_threads(threads)
            random.setstate(saved[0])
            np.random.set_state(saved[1])
            torch.set_rng_state(saved[2])
        if not pool.formulas:
            raise RuntimeError("Formula search found no valid formula; raise total_timesteps")
        self.formulas_ = list(pool.formulas)
        self.fills_ = [f[0] for f in pool.fills]
        self.ranges_ = [f[1:] for f in pool.fills]
        self.weights_ = pool.weights.copy()
        self.ic_, self.invalid_share_, self.episodes_ = pool.ic, halves, env.episodes
        self.timesteps_ = rollouts * n_steps
        logger.info("Formula search: %d episodes, pool IC %.4f, formulas %s", env.episodes, pool.ic, self.formulas_)
        return self

    def transform(self, windows):
        windows = np.asarray(windows, dtype=np.float32)
        columns = [np.clip(evaluate(f, windows, self.channels, fill=fill), *bounds)
                   for f, fill, bounds in zip(self.formulas_, self.fills_, self.ranges_)]
        return np.column_stack(columns).astype(np.float32)
