"""A trained run scores new rows later exactly as training scored its test rows.

Each run is trained in a subprocess on a tiny synthetic project, and every scoring runs in a subprocess too: earlier
test files may have loaded torch into this process, after which LightGBM can crash (two OpenMP runtimes).
"""

import json
import os
import pickle
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import polars as pl
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
ENV = {**os.environ, "ALPHAPY_ROOT": str(ROOT), "PYTHONPATH": os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")])}
TIGHT = {"rtol": 1e-9, "atol": 1e-12}


def rows(n, seed, start="2020-01-01"):
    """Numeric columns with gaps, a positive skewed column, integer hour and weekday columns, a factor and a text
    column; the label depends on ``a`` and the hour."""
    rng = np.random.default_rng(seed)
    a = rng.normal(size=n)
    hour = rng.integers(0, 24, n)
    frame = pd.DataFrame({
        "date": pd.date_range(start, periods=n, freq="D").strftime("%Y-%m-%d"),
        "a": a, "b": rng.normal(size=n), "c": rng.normal(loc=3.0, scale=2.0, size=n),
        "volume": rng.lognormal(mean=8.0, sigma=1.0, size=n),
        "hour": hour, "day_of_week": rng.integers(0, 7, n),
        "sector": rng.choice(["tech", "energy", "retail"], n),
        "venue": rng.choice(["x", "y", "z"], n),
        "label": (a + 0.08 * (hour - 12) + 0.7 * rng.normal(size=n) > 0).astype(int),
    })
    frame.loc[rng.random(n) < 0.06, "a"] = np.nan
    frame.loc[rng.random(n) < 0.04, "c"] = np.nan
    return frame


def tabular_config(algorithms, **features):
    config = yaml.safe_load((ROOT / "config/model.yml.template").read_text())
    config["project"] |= {"directory": ".", "file_extension": "csv", "live_results": False, "submission_file": None,
                          "submit_probas": False}
    config["model"] |= {"algorithms": algorithms, "target": "label", "estimators": 21, "cv_folds": 3,
                        "grid_search": {"option": False}, "scoring_function": "roc_auc",
                        "calibration": {"option": True, "type": "isotonic"},
                        "cv_time_blocks": {"option": True, "column": "date", "embargo": 1}}
    config["data"] |= {"drop": ["date"], "shuffle": False}
    config["features"] |= {
        "scaling": {"option": True, "type": "standard"},
        "counts": {"option": True},
        "factors": ["sector"],
        "encoding": {"rounding": 3, "type": "target"},
        "clustering": {"option": True, "increment": 2, "maximum": 4, "minimum": 2},
        "pca": {"option": True, "increment": 1, "maximum": 3, "minimum": 2, "whiten": False},
        "numpy": {"option": True},
        "interactions": {"option": True, "poly_degree": 2, "sampling_pct": 50},
        "variance": {"option": True, "threshold": 0.0},
        "log_transform": {"option": True, "pvalue_level": 0.05},
    } | features
    config["plots"] = {k: False for k in config["plots"]}
    config["pipeline"]["verbosity"] = 0
    return config


def make_project(folder: Path, config: dict, train: pd.DataFrame, test: pd.DataFrame) -> Path:
    (folder / "config").mkdir(parents=True)
    (folder / "data").mkdir()
    (folder / "config/model.yml").write_text(yaml.safe_dump(config, sort_keys=False))
    train.to_csv(folder / "data/train.csv", index=False)
    test.to_csv(folder / "data/test.csv", index=False)
    return folder


def run_alphapy(folder: Path, *args, fails=False) -> str:
    done = subprocess.run([sys.executable, "-c", "import sys; sys.argv = ['alphapy', *sys.argv[1:]]; "
                           "from alphapy.alphapy_main import main; main()", *args],
                          cwd=folder, env=ENV, capture_output=True, text=True, timeout=600)
    log = done.stdout + done.stderr
    assert (done.returncode != 0) == fails, log[-4000:]
    return log


def train(folder: Path) -> Path:
    run_alphapy(folder)
    return sorted((folder / "runs").iterdir())[-1]


def score(run: Path, frame: Path, *, mode="batch", history=None, algorithms=None, pandas=False, short=False):
    """Score a CSV of raw rows through ``alphapy.inference`` in a subprocess (read as training reads its files, or
    with pandas and a labelled index). ``mode`` is ``batch`` (one call), ``rows`` (one call per row) or ``features``
    (the feature matrix); ``history`` is None, ``"tail"`` (the training tail) or ``"follows"``
    (``history_from_training``); ``short`` allows short window history."""
    out = frame.with_suffix(f".{mode}.pkl")
    code = textwrap.dedent(f"""
        import pickle, sys
        import pandas as pd, polars as pl
        import alphapy
        from alphapy.frame import read_frame
        from alphapy.inference import load_run
        assert alphapy.__file__.startswith({str(ROOT)!r}), alphapy.__file__
        run = load_run({str(run)!r})
        frame = read_frame({str(frame.parent)!r}, {frame.stem!r}, 'csv', ',')
        if {pandas!r}:
            frame = pd.read_csv({str(frame)!r}).set_axis([f'row{{i}}' for i in range(frame.height)])
        history = {{None: None, 'tail': run.training_tail,
                    'follows': run.history_from_training(frame) if {history!r} == 'follows' else None}}[{history!r}]
        if {mode!r} == 'batch':
            result = run.predict_proba(frame, algorithms={algorithms!r}, history=history,
                                       allow_short_history={short!r})
        elif {mode!r} == 'rows':
            result = pd.concat([run.predict_proba(frame[i:i + 1], algorithms={algorithms!r}, history=history)
                                .set_axis([i]) for i in range(frame.height)])
        else:
            result = run.features(frame)
        pickle.dump(result, open({str(out)!r}, 'wb'))
    """)
    done = subprocess.run([sys.executable, "-c", code], env=ENV, capture_output=True, text=True, timeout=600)
    assert done.returncode == 0, (done.stdout + done.stderr)[-4000:]
    with open(out, "rb") as f:
        return pickle.load(f)


def training_probabilities(run: Path, partition="test") -> pd.DataFrame:
    """Training's probabilities by date (the ranked file is sorted by probability, so never compare by position)."""
    ranked = pd.read_csv(run / f"output/ranked_{partition}.csv")
    return ranked.set_index("date").sort_index()


def assert_reproduces(scores: pd.DataFrame, run: Path, frame: pd.DataFrame, partition="test"):
    manifest = json.loads((run / "model/manifest.json").read_text())
    expected = training_probabilities(run, partition)
    scores = scores.set_axis(frame["date"].to_numpy()).sort_index()
    assert list(scores.index) == list(expected.index)
    columns = manifest["algorithms"] + (["blend"] if manifest["blend"] else [])
    assert list(scores.columns) == columns
    for column in columns:
        recorded = expected[f"prob_{partition}_{column.lower()}"].to_numpy()
        assert np.allclose(scores[column].to_numpy(), recorded, **TIGHT), column


@pytest.fixture(scope="module")
def tabular(tmp_path_factory):
    folder = tmp_path_factory.mktemp("tabular")
    train_rows, test_rows = rows(900, 1), rows(300, 2, start="2022-06-19")
    run = train(make_project(folder, tabular_config(["LGB", "LOGR"]), train_rows, test_rows))
    return folder, run, test_rows


def test_predict_proba_reproduces_the_test_probabilities_of_every_algorithm_and_the_blend(tabular):
    folder, run, test_rows = tabular
    assert_reproduces(score(run, folder / "data/test.csv"), run, test_rows)


def test_predict_mode_writes_the_test_probabilities_training_recorded(tabular):
    folder, run, test_rows = tabular
    recorded = (run / "output/ranked_test.csv").read_bytes()
    run_alphapy(folder, "--predict", "--rundir", run.name)                      # a run name under runs/
    assert (run / "output/ranked_test.csv").read_bytes() == recorded          # training's record is kept
    predicted = pd.read_csv(run / "output/ranked_predict.csv")
    expected = pd.read_csv(run / "output/ranked_test.csv")
    assert list(predicted.columns) == list(expected.columns)                    # the same ranked format
    assert predicted["prob_test_best"].is_monotonic_decreasing
    predicted, expected = predicted.set_index("date").sort_index(), expected.set_index("date").sort_index()
    for column in [c for c in expected.columns if c.startswith("prob_test_")]:
        assert np.allclose(predicted[column], expected[column], **TIGHT), column
    for column in [c for c in expected.columns if c.startswith("pred_test_")]:
        assert (predicted[column] == expected[column]).all(), column


def test_scoring_rows_one_at_a_time_equals_scoring_them_together(tabular):
    folder, run, test_rows = tabular
    test_rows.iloc[:40].to_csv(folder / "first40.csv", index=False)
    together = score(run, folder / "first40.csv")
    one_by_one = score(run, folder / "first40.csv", mode="rows")
    assert np.allclose(one_by_one.to_numpy(), together.to_numpy(), **TIGHT)


def test_a_single_row_with_zero_hour_and_weekday_and_a_gap_keeps_every_feature(tabular):
    folder, run, test_rows = tabular
    row = test_rows.iloc[[5]].assign(hour=0, day_of_week=0, a=np.nan)
    row.to_csv(folder / "one.csv", index=False)
    pd.concat([row, test_rows.iloc[:30]]).to_csv(folder / "with_others.csv", index=False)
    features = score(run, folder / "one.csv", mode="features")
    manifest = json.loads((run / "model/manifest.json").read_text())
    assert list(features.columns) == manifest["feature_names"] and len(features) == 1
    alone = score(run, folder / "one.csv")
    among = score(run, folder / "with_others.csv")
    assert np.allclose(alone.to_numpy(), among.iloc[:1].to_numpy(), **TIGHT)


def test_a_pandas_frame_keeps_its_index_and_a_gap_in_an_integer_column_is_imputed(tabular):
    """pandas reads an integer column with a gap as floats; the run casts it back and imputes as training did."""
    folder, run, test_rows = tabular
    test_rows.iloc[:20].assign(hour=lambda d: d["hour"].where(d.index != d.index[3])).to_csv(folder / "gap.csv",
                                                                                          index=False)
    from_pandas = score(run, folder / "gap.csv", pandas=True)
    from_polars = score(run, folder / "gap.csv")
    assert list(from_pandas.index) == [f"row{i}" for i in range(20)]
    assert np.allclose(from_pandas.to_numpy(), from_polars.to_numpy(), **TIGHT)


def test_no_fitted_feature_step_reads_the_test_rows(tabular, tmp_path):
    """The same training rows with wildly different test rows give the same model: every training probability and
    the fitted pipeline are unchanged. Training used to fit imputation, scaling, clustering, PCA and the low-variance
    mask on training and test rows together."""
    folder, run, test_rows = tabular
    changed = test_rows.assign(a=test_rows["a"] * 1000 + 50, b=np.nan, volume=test_rows["volume"] * 1e6,
                               hour=23, sector="unseen", venue="never")
    other = train(make_project(tmp_path / "other", tabular_config(["LGB", "LOGR"]),
                               pd.read_csv(folder / "data/train.csv"), changed))
    first, second = training_probabilities(run, "train"), training_probabilities(other, "train")
    for column in ("prob_train_lgb", "prob_train_logr", "prob_train_blend"):
        assert np.allclose(first[column], second[column], **TIGHT), column
    # and both pipelines turn the same rows into the same features
    assert np.allclose(score(run, folder / "data/test.csv", mode="features").to_numpy(),
                       score(other, folder / "data/test.csv", mode="features").to_numpy(), **TIGHT)


def test_a_frame_missing_a_training_column_is_refused(tabular):
    folder, run, test_rows = tabular
    test_rows.drop(columns=["hour"]).iloc[:5].to_csv(folder / "no_hour.csv", index=False)
    with pytest.raises(AssertionError, match=r"lacks columns the run was trained on: \['hour'\]"):
        score(run, folder / "no_hour.csv")


def copy_project(folder: Path, run: Path, to: Path) -> Path:
    shutil.copytree(folder, to)
    return to / "runs" / run.name


def test_predict_mode_scores_unlabelled_rows(tabular, tmp_path):
    folder, run, test_rows = tabular
    run = copy_project(folder, run, tmp_path / "copy")
    test_rows.assign(label=test_rows["label"].where(test_rows.index % 3 != 0)).to_csv(tmp_path / "copy/data/test.csv",
                                                                                     index=False)
    run_alphapy(tmp_path / "copy", "--predict", "--rundir", str(run))
    predicted = pd.read_csv(run / "output/ranked_predict.csv")
    assert len(predicted) == len(test_rows) and predicted["label"].isna().sum() == 100
    expected = training_probabilities(run)
    assert np.allclose(predicted.set_index("date").sort_index()["prob_test_blend"], expected["prob_test_blend"],
                       **TIGHT)


def test_predict_mode_refuses_the_run_input_copy_and_live_results(tabular, tmp_path):
    folder, run, test_rows = tabular
    run = copy_project(folder, run, tmp_path / "copy")
    (tmp_path / "copy/data/test.csv").unlink()
    log = run_alphapy(tmp_path / "copy", "--predict", "--rundir", str(run), fails=True)
    assert "The project has no data/test.csv, and the run's input/test.csv is training's copy" in log
    test_rows.to_csv(tmp_path / "copy/data/test.csv", index=False)
    config = yaml.safe_load((tmp_path / "copy/config/model.yml").read_text())
    config["project"]["live_results"] = True
    (tmp_path / "copy/config/model.yml").write_text(yaml.safe_dump(config, sort_keys=False))
    (run / "output/ranked_predict.csv").unlink(missing_ok=True)          # left by an earlier predict-mode test
    log = run_alphapy(tmp_path / "copy", "--predict", "--rundir", str(run), fails=True)
    assert "live_results is set" in log and not (run / "output/ranked_predict.csv").exists()


def pipeline_for(train_rows, **specs):
    """A feature pipeline fitted in this process on ``train_rows`` (every option off unless given)."""
    from alphapy.features import PIPELINE_SPECS, FeaturePipeline
    from alphapy.globals import ModelType
    specs = {key: False for key in PIPELINE_SPECS} | {
        "drop": [], "factors": [], "sentinel": -1, "transforms": None, "model_type": ModelType.classification,
        "pvalue_level": 0.05, "lv_threshold": 0.0, "seed": 0, "n_jobs": 1} | specs
    frame = pl.from_pandas(train_rows)
    pipeline = FeaturePipeline(specs, frame.schema)
    model = SimpleNamespace(specs=specs, feature_map={}, algolist=[], fnames_algo={}, feature_names=[])
    pipeline.fit_transform(model, frame, np.arange(len(train_rows)) % 2)
    return pipeline


def test_a_text_value_training_never_saw_gets_minus_one():
    pipeline = pipeline_for(pd.DataFrame({"venue": ["x", "y", "x", "y"], "a": [1.0, 2.0, 3.0, 4.0]}))
    features = pipeline.transform(pipeline.conform(pd.DataFrame({"venue": ["z", "y"], "a": [1.0, 2.0]})))
    assert pipeline.feature_names == ["venue_factor", "a"] and features[:, 0].tolist() == [-1, 1]


def test_a_value_at_or_below_zero_in_a_log_transformed_column_is_refused():
    volume = np.random.default_rng(0).lognormal(8.0, 1.0, 200)
    pipeline = pipeline_for(pd.DataFrame({"volume": volume}), log_transform=True)
    assert pipeline.feature_names == ["volume_log"]
    with pytest.raises(ValueError, match="log-transformed in training"):
        pipeline.transform(pipeline.conform(pd.DataFrame({"volume": [100.0, 0.0]})))


def test_t_sne_is_refused():
    with pytest.raises(ValueError, match="t-SNE cannot place rows"):
        pipeline_for(pd.DataFrame({"a": np.arange(20.0), "b": np.arange(20.0) ** 2}), tsne=True)


def test_a_fractional_value_in_an_integer_column_is_refused():
    pipeline = pipeline_for(pd.DataFrame({"hour": [0, 5, 9, 23], "a": [1.0, 2.0, 3.0, 4.0]}))
    assert pipeline.conform(pd.DataFrame({"hour": [7.0, np.nan], "a": [1.0, 2.0]}))["hour"].to_list() == [7, None]
    with pytest.raises(ValueError, match=r"fractional values in columns training read as integers: \['hour'\]"):
        pipeline.conform(pd.DataFrame({"hour": [1.5], "a": [1.0]}))


def test_a_run_without_a_saved_feature_pipeline_is_refused(tmp_path):
    from alphapy.inference import load_run
    (tmp_path / "model").mkdir()
    (tmp_path / "model/manifest.json").write_text(json.dumps({"algorithms": ["LOGR"]}))
    with pytest.raises(FileNotFoundError, match="feature pipeline"):
        load_run(tmp_path)


def test_lofo_on_one_algorithm_scores_each_algorithm_with_its_own_columns(tmp_path):
    rng = np.random.default_rng(7)
    data = rows(1000, 3)
    for k in range(4):
        data[f"noise{k}"] = rng.normal(size=len(data))
    config = tabular_config(["LGB", "LOGR"], lofo={"option": True, "algorithms": ["LGB"]},
                            scaling={"option": True, "type": "minmax"},
                            interactions={"option": False, "poly_degree": 2, "sampling_pct": 50},
                            clustering={"option": False, "increment": 2, "maximum": 4, "minimum": 2},
                            pca={"option": False, "increment": 1, "maximum": 3, "minimum": 2, "whiten": False})
    test_rows = data.iloc[750:]
    run = train(make_project(tmp_path / "lofo", config, data.iloc[:750], test_rows))
    feature_map = joblib.load(run / "model/feature_map.pkl")
    support = feature_map[("support_lofo", "LGB")]
    assert ("support_lofo", "LOGR") not in feature_map and 0 < sum(support) < len(support)
    lgb, logr = (joblib.load(run / f"model/estimators/{a}.pkl") for a in ("LGB", "LOGR"))
    assert lgb.n_features_in_ == sum(support) < logr.n_features_in_
    assert_reproduces(score(run, tmp_path / "lofo/data/test.csv"), run, test_rows)


def test_scipy_and_isomap_features_reproduce_the_test_probabilities(tmp_path):
    """After standard scaling some row values are negative, so the geometric mean is missing in every row; the
    column is kept (zero), not dropped, so the names stay aligned with the columns."""
    config = tabular_config(["LGB", "LOGR"], scipy={"option": True},
                            isomap={"option": True, "components": 2, "neighbors": 5},
                            interactions={"option": False, "poly_degree": 2, "sampling_pct": 50})
    test_rows = rows(200, 5, start="2022-06-19")
    run = train(make_project(tmp_path / "scipy", config, rows(600, 4), test_rows))
    names = json.loads((run / "model/manifest.json").read_text())["feature_names"]
    assert "sp_signal_to_noise" in names and "isomap_2" in names
    assert_reproduces(score(run, tmp_path / "scipy/data/test.csv"), run, test_rows)


def bars(seed):
    """Two tickers whose each return reverses the last one, so a window predicts the next bar, plus a noise column."""
    rng = np.random.default_rng(seed)
    frames = []
    for ticker in ("AAA", "BBB"):
        r = np.zeros(500)
        for t in range(1, 500):
            r[t] = -0.8 * r[t - 1] + 0.003 * rng.normal()
        close = 100 * np.exp(np.cumsum(r))
        f = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=500, freq="D").strftime("%Y-%m-%d"),
                          "ticker": ticker, "close": close, "volume": rng.integers(1000, 2000, 500).astype(float),
                          "noise": rng.normal(size=500)})
        f["label"] = (f["close"].shift(-1) > f["close"]).astype(int)
        frames.append(f.iloc[:-1])
    return pd.concat(frames).sort_values(["date", "ticker"]).reset_index(drop=True)


def test_a_window_algorithm_scores_its_test_rows_as_training_did(tmp_path):
    pytest.importorskip("aeon")
    data = bars(0)
    cut = data["date"] >= "2021-01-01"
    config = tabular_config(["LOGR", "MROCK"], counts={"option": False}, factors=[],
                            interactions={"option": False, "poly_degree": 2, "sampling_pct": 50},
                            clustering={"option": False, "increment": 2, "maximum": 4, "minimum": 2},
                            pca={"option": False, "increment": 1, "maximum": 3, "minimum": 2, "whiten": False},
                            numpy={"option": False}, log_transform={"option": False, "pvalue_level": 0.05})
    config["data"]["drop"] = ["date", "ticker", "close", "volume"]
    config["sequence"] = {"option": True, "group": "ticker", "order": "date", "channels": ["close", "volume"],
                          "window": 12, "scale": {"close": "close", "volume": "mean"}}
    folder = make_project(tmp_path / "windows", config, data[~cut], data[cut])
    run = train(folder)
    test_rows = data[cut]
    with pytest.raises(AssertionError, match="have fewer than 11 earlier rows"):
        score(run, folder / "data/test.csv")                        # no history: the first rows' windows are short
    padded = score(run, folder / "data/test.csv", short=True)
    assert not np.allclose(padded["MROCK"].to_numpy()[:2], score(run, folder / "data/test.csv",
                                                                 history="tail")["MROCK"].to_numpy()[:2])
    data[~cut].iloc[-100:].to_csv(folder / "late_training.csv", index=False)   # not after training's last rows
    with pytest.raises(AssertionError, match="have fewer than 11 earlier rows"):
        score(run, folder / "late_training.csv", history="follows")
    scores = score(run, folder / "data/test.csv", history="follows")
    expected = pd.read_csv(run / "output/ranked_test.csv").set_index(["date", "ticker"]).sort_index()
    scores = scores.set_axis(pd.MultiIndex.from_frame(test_rows[["date", "ticker"]])).sort_index()
    for column in ("LOGR", "MROCK", "blend"):
        assert np.allclose(scores[column], expected[f"prob_test_{column.lower()}"], **TIGHT), column
    run_alphapy(folder, "--predict", "--rundir", str(run))
    predicted = pd.read_csv(run / "output/ranked_predict.csv").set_index(["date", "ticker"]).sort_index()
    for column in ("logr", "mrock", "blend"):
        assert np.allclose(predicted[f"prob_test_{column}"], expected[f"prob_test_{column}"], **TIGHT), column
