"""TS2Vec: the loss behaves as a contrastive loss, and its label-free features expose a shape a classifier can use."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from alphapy.ts2vec import TS2VecLearner, hierarchical_loss  # noqa: E402


def test_matching_views_score_lower_than_unrelated_ones():
    from alphapy.networks import torch_threads
    with torch_threads(1):              # torch on the CPU after LightGBM in one process: one thread (alphapy.networks)
        g = torch.Generator().manual_seed(0)
        z = torch.nn.functional.normalize(torch.randn(16, 8, 32, generator=g), dim=-1) * 5
        matching = hierarchical_loss(z, z + 0.01 * torch.randn(z.shape, generator=g))
        unrelated = hierarchical_loss(z, torch.randn(z.shape, generator=g) * 5)
    assert matching < unrelated


def planted(n, seed):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    shape = 1 - 0.1 * np.sin(np.linspace(0, np.pi, 20))
    close = np.where(y[:, None] == 1, shape, 2 - shape) + 0.04 * rng.normal(size=(n, 20))
    return np.stack([close, np.ones((n, 20))], axis=1).astype(np.float32), y


def test_label_free_features_expose_the_shape_and_each_row_is_encoded_on_its_own():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    w, y = planted(600, 0)
    w_new, y_new = planted(300, 1)
    learner = TS2VecLearner(hidden=32, out=32, depth=3, epochs=5, device="cpu").fit(w)       # no labels
    clf = LogisticRegression(max_iter=2000).fit(learner.transform(w), y)
    assert roc_auc_score(y_new, clf.predict_proba(learner.transform(w_new))[:, 1]) > 0.8
    alone = learner.transform(w_new[:1])
    assert np.allclose(learner.transform(w_new)[:1], alone, atol=1e-5)
