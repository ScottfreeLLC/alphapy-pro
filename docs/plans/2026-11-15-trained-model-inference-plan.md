# Inference from trained runs (2026-11-15)

## Goal

A trained AlphaPy run must be reusable: score new rows later with every fitted model and the blend, exactly as
training scored its test rows, from the command line (`alphapy --predict`) and from Python in-process (a frame of
recent rows, for paper studies in alphapy-finance).

## Findings (2026-11-15)

- **Predict mode fails on a freshly trained run** (checked by running it on a two-model project). The run writes
  `model/model.pkl` and `model/feature_map.pkl`, but predict mode searches for `model_*` and `feature_map_*`
  (`alphapy/model.py:566, 652`). It logs "Could not find feature map" and then raises `ValueError: max() iterable
  argument is empty`. No test covers predict mode.
- **Only the best single model is saved** (`save_predictor`, `model.py:589-624`). The other fitted algorithms and
  the blend are lost; the next-bar crypto run kept only its calibrated random forest.
- **Feature creation fits on the whole batch.** `create_features` (`alphapy/features.py:1052`) and its helpers fit
  these on training and test rows together:
  - imputation medians;
  - the log-transform test;
  - `StandardScaler` / `MinMaxScaler`;
  - text vectorizers;
  - clustering, PCA, isomap and t-SNE.
  So training reads the test rows' statistics (a look-ahead), and a new row cannot be transformed as training
  did. The next-bar configs have standard scaling on.
- **Predict mode slices one feature support for every model.** Training keeps per-algorithm supports (LOFO on LGB
  only, univariate, RFE) through `apply_feature_selection_support`. The sequence section (window models) is
  refused.

## Design

- **Fit on training rows, save, apply.** Every step that learns from data is fitted on the training rows only,
  saved in the run as a fitted object, and applied unchanged to test rows and to any rows scored later. This covers
  imputation medians, the log-transform decision, scalers, encoders, vectorizers, clustering, PCA, isomap and
  t-SNE, the low-variance column mask and the feature supports. Training results change for configurations that
  use these steps (no more test-row statistics); that is the intended fix.
- **Save the whole run.** In `model/`:
  - `model.pkl` (the best model, unchanged: alphapy-finance reads it);
  - every fitted algorithm (`estimators.pkl`, keyed by algorithm);
  - the blend's members;
  - the fitted feature pipeline;
  - the feature map with per-algorithm supports;
  - the sequence spec;
  - a manifest (alphapy version, algorithms, best, feature names, tabular column count).
- **One scoring function.** `alphapy.inference.load_run(run_dir)` returns an object whose
  `predict_proba(frame, algorithms=None)` gives each algorithm's and the blend's probabilities for the frame's rows:
  - it applies the saved pipeline;
  - it applies each algorithm's own support through the same gate as training;
  - for window algorithms, it builds windows from the frame's history (the frame carries enough earlier rows per
    group).
  Predict mode uses this function and writes the same ranked output as training.

## Steps

The order matters: save every model first, on the current feature code, so a refit can be checked against the
recorded probabilities before feature fitting changes.

1. **Save every fitted algorithm** (one file per algorithm beside the unchanged `model.pkl` and `feature_map.pkl`),
   the blend's members (the blend is a plain average, `predict_blend`) and a manifest. Measure the saved size on
   the crypto 15-minute cell before any million-row cell. Status: done. `save_estimators` writes
   `model/estimators/<ALGO>.pkl` and `model/manifest.json`; `tests/test_saved_run.py`. Crypto 15-minute: 131 MB for
   the five models (the random forest dominates).
2. **Refit check (alphapy-finance).**
   - Rerun the crypto 15-minute calibrated cell with the same config and seed.
   - Compare every algorithm's and the blend's held-back probabilities with the recorded `ranked_test.csv`.
   - If they don't reproduce, report that the validated blend cannot be recovered and any artifact is a new fit.
   Status: done. `projects/next-bar/crypto/15min_calibrated_lofo_refit/runs/run_20261115_210812` (114 s, 5.7 GB peak)
   against `run_20261114_185343` on all 130,686 held-back rows. LightGBM, XGBoost, CatBoost and logistic regression
   are identical; random forest differs by at most 6.7e-13 and the blend by at most 1.3e-13. The P(up) >= 0.55
   decision is the same on every row. The validated blend is recovered and now saved.
3. **Tests first:**
   - a trained two-model run reproduces its test probabilities through predict mode and through `predict_proba`,
     for each algorithm and the blend;
   - one-at-a-time equals batch scoring;
   - a one-row frame with `hour` 0, `day_of_week` 0 and a null keeps every column;
   - a test fails if any step reads test rows when fitting.
   Status: pending.
4. **Fit on training rows and save the fitted pipeline.**
   - Covers imputation medians, the log-transform decision, scalers, encoders, vectorizers, clustering, PCA,
     isomap, t-SNE and the low-variance mask.
   - Save training's column names, order, dtypes and keep or drop decisions. `create_features` drops all-zero
     columns and routes columns by dtype and number of unique values, both of which a small frame changes. Apply
     these by name and refuse a frame that does not match.
   - Training results change wherever scaling or imputation is on. That is the intended fix, and it is stated in
     the pull request.
   Status: pending.
5. **`alphapy.inference`** (`load_run`, `predict_proba`): per-algorithm supports through the training gate; window
   algorithms fed the training tail as history, as in training. Status: pending.
6. **Predict mode on top of it** (fix the file names; sequence supported). Status: pending.
7. **Close out.**
   - Full alphapy-pro suite.
   - alphapy-finance `tests/systems` and ML tests with a thread timeout (it imports this checkout editable).
   - Restart its server, Fable review, pull request to `main`.
   - Rerunning the crypto cell after step 4 measures the look-ahead. That rerun scores held-back rows with a
     changed model, so its trial line is written in the Record before it runs.
   Status: pending.

## Effect on alphapy-finance

- The next-bar study's AlphaPy runs fitted scaling and imputation on development and held-back rows together. That
  is a look-ahead, to be recorded in its pre-registration Record.
- Rerunning a cell after step 2 shows its size.
- The paper-study plan (alphapy-finance `docs/plans/2026-11-15-004-feat-next-bar-crypto-paper-study-plan.md`,
  step 3) uses `alphapy.inference` instead of its own copy of the feature steps.
