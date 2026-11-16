"""A trained run saves every fitted algorithm and a manifest, so the run can be scored again later."""

import json
import os
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def rows(n=1200, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 3))
    return pd.DataFrame({"date": pd.date_range("2020-01-01", periods=n, freq="D").strftime("%Y-%m-%d"),
                         "a": x[:, 0], "b": x[:, 1], "c": x[:, 2],
                         "label": (x[:, 0] + 0.5 * rng.normal(size=n) > 0).astype(int)})


def project(tmp_path: Path, algorithms: list[str]) -> Path:
    (tmp_path / "config").mkdir()
    (tmp_path / "data").mkdir()
    data = rows()
    data.iloc[:900].to_csv(tmp_path / "data/train.csv", index=False)
    data.iloc[900:].to_csv(tmp_path / "data/test.csv", index=False)
    config = yaml.safe_load((ROOT / "config/model.yml.template").read_text())
    config["project"] |= {"directory": ".", "file_extension": "csv", "live_results": False, "submission_file": None,
                          "submit_probas": False}
    config["model"] |= {"algorithms": algorithms, "target": "label", "estimators": 21, "cv_folds": 3,
                        "cv_time_blocks": {"option": True, "column": "date", "embargo": 1}}
    config["data"]["drop"] = ["date"]
    config["plots"] = {k: False for k in config["plots"]}
    (tmp_path / "config/model.yml").write_text(yaml.safe_dump(config, sort_keys=False))
    return tmp_path


def train(folder: Path) -> Path:
    env = {**os.environ, "ALPHAPY_ROOT": str(ROOT)}
    done = subprocess.run([sys.executable, "-m", "alphapy.alphapy_main"], cwd=folder, env=env,
                          capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    return sorted((folder / "runs").iterdir())[-1]


def test_every_fitted_algorithm_and_the_blend_members_are_saved(tmp_path):
    run = train(project(tmp_path, ["LGB", "LOGR"]))
    manifest = json.loads((run / "model/manifest.json").read_text())
    assert manifest["algorithms"] == ["LGB", "LOGR"] and manifest["blend"] == ["LGB", "LOGR"]
    assert manifest["best_algorithm"] in {"LGB", "LOGR", "BLEND"}
    assert manifest["feature_names"] and manifest["target"] == "label"
    for algo in manifest["algorithms"]:
        estimator = joblib.load(run / f"model/estimators/{algo}.pkl")
        assert hasattr(estimator, "predict_proba")
        assert estimator.n_features_in_ == len(manifest["feature_names"])
    assert (run / "model/model.pkl").exists() and (run / "model/feature_map.pkl").exists()   # unchanged contract


def test_a_single_algorithm_run_has_no_blend(tmp_path):
    run = train(project(tmp_path, ["LOGR"]))
    manifest = json.loads((run / "model/manifest.json").read_text())
    assert manifest["algorithms"] == ["LOGR"] and manifest["blend"] == []


def test_a_regression_blend_is_the_mean_not_rounded_and_the_manifest_records_library_versions(tmp_path):
    from types import SimpleNamespace
    from alphapy.globals import ModelType, Partition
    from alphapy.model import predict_blend
    model = SimpleNamespace(specs={"model_type": ModelType.regression},
                            preds={("A", Partition.train): np.array([1.2, 2.6]), ("B", Partition.train): np.array([1.4, 2.9]),
                                   ("A", Partition.test): np.array([0.3]), ("B", Partition.test): np.array([0.4])},
                            probas={})
    predict_blend(model)
    assert np.allclose(model.preds[("BLEND", Partition.train)], [1.3, 2.75])
    assert np.allclose(model.preds[("BLEND", Partition.test)], [0.35])
    run = train(project(tmp_path, ["LOGR"]))
    libraries = json.loads((run / "model/manifest.json").read_text())["libraries"]
    assert {"scikit-learn", "numpy"} <= set(libraries)
