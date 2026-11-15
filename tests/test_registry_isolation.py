"""Window algorithms cost nothing to projects that do not use them: torch is not loaded, and LightGBM's multithreaded
cross-validation (which segfaulted once torch was loaded at start-up) still runs."""

import subprocess
import sys

import pytest

pytest.importorskip("lightgbm")

SCRIPT = """
import sys
import numpy as np
from alphapy.estimators import find_optional_packages, estimator_map, WINDOW_ALGOS
find_optional_packages()
assert "torch" not in sys.modules, "torch was imported at start-up"
assert not set(WINDOW_ALGOS) & set(estimator_map), "window algorithms are offered without the sequence section"
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
rng = np.random.default_rng(0)
X, y = rng.normal(size=(3000, 10)), rng.integers(0, 2, 3000)
print(cross_val_score(LGBMClassifier(n_estimators=30, n_jobs=-1, verbose=-1), X, y, cv=3, n_jobs=-1).mean())
"""


def test_registry_loads_no_window_dependency_and_multithreaded_lightgbm_cross_validation_runs():
    run = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr[-2000:]
