"""The whole AlphaPy pipeline with the sequence section on: tabular and window algorithms side by side, calibration
over time-ordered folds and SHAP, on bars whose next move depends on the window's shape (mean reversion)."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.metrics import roc_auc_score

pytest.importorskip("torch")
pytest.importorskip("aeon")
pytest.importorskip("shap")

ROOT = Path(__file__).resolve().parents[1]


def bars(seed):
    """Two tickers; each return reverses the last one (so the window predicts the next), plus a noise feature."""
    rng = np.random.default_rng(seed)
    frames = []
    for ticker in ("AAA", "BBB"):
        r = np.zeros(900)
        for t in range(1, 900):
            r[t] = -0.8 * r[t - 1] + 0.003 * rng.normal()
        close = 100 * np.exp(np.cumsum(r))
        f = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=900, freq="D").strftime("%Y-%m-%d"),
                          "ticker": ticker, "open": close * (1 + 0.001 * rng.normal(size=900)),
                          "high": close * 1.002, "low": close * 0.998, "close": close,
                          "volume": rng.integers(1000, 2000, 900).astype(float), "noise": rng.normal(size=900)})
        f["label"] = (f["close"].shift(-1) > f["close"]).astype(int)
        frames.append(f.iloc[:-1])
    return pd.concat(frames).sort_values(["date", "ticker"]).reset_index(drop=True)


def config():
    return {
        "project": {"directory": ".", "file_extension": "csv", "live_results": False, "submission_file": None,
                    "submit_probas": False},
        "model": {"algorithms": ["LGB", "CNN", "MROCK"], "calibration": {"option": True, "type": "isotonic"},
                  "cv_folds": 3, "estimators": 51, "grid_search": False, "optuna_trials": 5, "optuna_timeout": 30,
                  "ranking": {"group_id": None, "group_size": None}, "pvalue_level": 0.01,
                  "rfe": {"option": False, "step": 3}, "scoring_function": "roc_auc", "target": "label",
                  "time_series": {"option": False, "date_index": "date", "forecast": 1, "n_lags": 0,
                                  "group_id": "unique_id", "leaders": []},
                  "type": "classification", "cv_time_blocks": {"option": True, "column": "date", "embargo": 1}},
        "data": {"allow_na_targets": False, "drop": ["date", "ticker", "open", "high", "low", "close", "volume"],
                 "features": "*", "leaders": [], "sampling": {"option": False, "method": "under_random", "ratio": 0.5},
                 "sentinel": -1, "separator": ",", "shuffle": False, "split": 0.3},
        "features": {"clustering": {"option": False, "increment": 3, "maximum": 30, "minimum": 3},
                     "counts": {"option": False}, "encoding": {"rounding": 3, "type": "target"}, "factors": [],
                     "interactions": {"option": False, "poly_degree": 2, "sampling_pct": 10},
                     "isomap": {"option": False, "components": 2, "neighbors": 5},
                     "lofo": {"option": False, "algorithms": ["LGB"]},
                     "log_transform": {"option": False, "pvalue_level": 0.01}, "numpy": {"option": False},
                     "pca": {"option": False, "increment": 1, "maximum": 10, "minimum": 2, "whiten": False},
                     "scaling": {"option": True, "type": "standard"}, "scipy": {"option": False},
                     "text": {"ngrams": 1, "vectorize": False},
                     "tsne": {"option": False, "components": 2, "learning_rate": 1000.0, "perplexity": 30.0},
                     "univariate": {"option": False, "percentage": 50, "uni_grid": [5, 10],
                                    "score_func": "f_regression"},
                     "variance": {"option": False, "threshold": 0.1}},
        "pipeline": {"number_jobs": -1, "seed": 42, "verbosity": 0},   # the setting that segfaulted LightGBM
        "plots": {"calibration": False, "confusion_matrix": False, "importances": False, "learning_curve": False,
                  "roc_curve": False, "shap": True, "shap_samples": 200},
        "sequence": {"option": True, "group": "ticker", "order": "date", "channels": ["close", "volume"],
                     "window": 12, "scale": {"close": "close", "volume": "mean"}},
        "xgboost": {"stopping_rounds": 30},
    }


def test_tabular_and_window_algorithms_run_side_by_side_through_the_pipeline(tmp_path):
    data = bars(0)
    cut = data["date"] >= "2022-01-01"
    (tmp_path / "config").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / "config" / "model.yml").write_text(yaml.safe_dump(config(), sort_keys=False))
    data[~cut].to_csv(tmp_path / "data" / "train.csv", index=False)
    data[cut].to_csv(tmp_path / "data" / "test.csv", index=False)
    run = subprocess.run([sys.executable, "-c", "import sys; sys.argv = ['alphapy']; "
                          "from alphapy.alphapy_main import main; main()"],
                         cwd=tmp_path, env={**os.environ, "ALPHAPY_ROOT": str(ROOT)}, capture_output=True,
                         text=True, timeout=900)
    assert run.returncode == 0, run.stderr[-3000:]
    log = run.stdout + run.stderr
    assert "Appended 36 window columns to 1 tabular columns" in log and "Segmentation" not in log
    out = next((tmp_path / "runs").iterdir()) / "output"
    ranked = pd.read_csv(out / "ranked_test.csv")
    auc = {a: roc_auc_score(ranked["label"], ranked[f"prob_test_{a}"]) for a in ("lgb", "cnn", "mrock")}
    assert auc["cnn"] > 0.65 and auc["mrock"] > 0.65, auc      # the windows carry the signal
    assert auc["lgb"] < 0.6, auc                               # the tabular noise column does not
    assert (out / "shap_ranking_test_LGB.csv").exists()         # tabular SHAP ran
    assert pd.read_csv(out / "shap_ranking_test_LGB.csv")["feature"].tolist() == ["noise"]   # no window columns
