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

1. Tests first:
   - a trained two-model run must reproduce its test probabilities through predict mode and through
     `predict_proba`, for each algorithm and the blend;
   - scoring rows one at a time must equal scoring them in a batch;
   - a test that fails if any step reads test rows when fitting.
   Status: pending.
2. Fit-on-train for every data-learning step in `features.py`; save the fitted pipeline. Status: pending.
3. Save every fitted algorithm, the blend, supports, the sequence spec and the manifest. Status: pending.
4. `alphapy.inference` (`load_run`, `predict_proba`), window algorithms included. Status: pending.
5. Predict mode on top of it (fix the file names; per-algorithm supports; sequence supported). Status: pending.
6. Full suite; Fable review; pull request to `main`. Status: pending.

## Effect on alphapy-finance

- The next-bar study's AlphaPy runs fitted scaling and imputation on development and held-back rows together. That
  is a look-ahead, to be recorded in its pre-registration Record.
- Rerunning a cell after step 2 shows its size.
- The paper-study plan (alphapy-finance `docs/plans/2026-11-15-004-feat-next-bar-crypto-paper-study-plan.md`,
  step 3) uses `alphapy.inference` instead of its own copy of the feature steps.
