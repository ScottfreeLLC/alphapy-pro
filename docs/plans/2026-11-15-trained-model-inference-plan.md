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
   Status: done (2026-11-15). `tests/test_inference.py`, every training and scoring in a subprocess. Also: LOFO on
   one algorithm only, a MiniRocket window run, a frame missing a column and a run without a saved pipeline. Before
   the change: the module did not exist, predict mode failed (`Could not find feature map`, then `KeyError:
   'lv_support'`), training with PCA failed the names check, and changing only the test rows moved training
   probabilities by up to 0.41 (LightGBM), 0.15 (logistic regression) and 0.20 (blend).
4. **Fit on training rows and save the fitted pipeline.**
   - Covers imputation medians, the log-transform decision, scalers, encoders, vectorizers, clustering, PCA,
     isomap, t-SNE and the low-variance mask.
   - Save training's column names, order, dtypes and keep or drop decisions. `create_features` drops all-zero
     columns and routes columns by dtype and number of unique values, both of which a small frame changes. Apply
     these by name and refuse a frame that does not match.
   - Training results change wherever scaling or imputation is on. That is the intended fix, and it is stated in
     the pull request.
   Status: done (2026-11-15). `alphapy.features.FeaturePipeline` replaces `create_features`,
   `create_interactions` and their helpers; training produces its test features with the same `transform` that
   scores later rows; saved as `model/feature_pipeline.pkl`. Findings and decisions:
   - Also fitted on training rows: the NumPy and SciPy row statistics' imputer and scaler, the interaction scaler,
     and the scalers inside clustering, PCA and isomap; text codes are training's (an unseen value gets -1).
   - t-SNE is refused in training: it has no transform, so it could score test rows only by fitting on them.
   - A value at or below zero in a column training log-transformed is refused (training's test rows included).
   - Already broken on develop and fixed: interactions (`get_feature_names` removed from scikit-learn), PCA (one
     name per block failed the names check), SciPy features (`signaltonoise` removed from SciPy). The `_log`
     suffix went on the columns that were not log-transformed; it now goes on those that are.
   - Text vectorization was never reached (`get_feature_names`), so text was always factorized; unchanged, with a
     warning when `vectorize` is on.
   - User transforms (`transforms`) still run on whatever rows they are given: on training and test rows together
     in training, on the frame alone at inference. A transform that reads across rows needs those rows present.
5. **`alphapy.inference`** (`load_run`, `predict_proba`): per-algorithm supports through the training gate; window
   algorithms fed the training tail as history, as in training. Status: done (2026-11-15).
   `load_run(run_dir).predict_proba(frame, algorithms=None, history=None)`, also `predict`, `score` and `features`.
   The frame is conformed to training's raw columns by name and types; extra columns (the target) are ignored.
   The training tail is passed explicitly (`history=run.training_tail`), never prepended by default: stale history
   would be wrong for rows that do not follow training's. RFE supports and each algorithm's feature names are now
   in the manifest. Runs trained before step 4, including the recovered crypto refit of step 2, have no saved
   pipeline and are refused; they must be trained again to be scored this way.
6. **Predict mode on top of it** (fix the file names; sequence supported). Status: done (2026-11-15). Writes
   `output/ranked_predict` in training's `ranked_test` format (best, blend, every algorithm) and keeps training's
   files; window algorithms read the training tail. `--rundir` accepts a path or a run name under `runs/`. The old
   `load_predictor` and `load_feature_map` are removed. alphapy-sports' `sflow_main` calls predict mode after
   writing its prediction frame to the run's `input/test.csv`; it would now get `ranked_predict` instead of a
   crash (not run against alphapy-sports here).
   Full alphapy-pro suite after steps 3-6: `118 passed, 33 warnings in 183.43s` (branch
   `feature/trained-model-inference`). The SciPy row statistics needed one more fix: after standard scaling the
   geometric mean is missing in every row and its column was dropped (8 columns, 9 names); it is now kept as
   zero. Not exercised through inference: RFE supports, regression runs.
   Review fixes (2026-11-15, after Fable's review):
   - A window algorithm's row with fewer than `window - 1` earlier rows in its group is refused, naming how many
     rows and which groups (a probe scored MROCK 0.512 without history against 0.195 with it);
     `allow_short_history=True` scores the padded window anyway. With a session column, only rows whose session may
     begin before the rows given count as short.
   - Predict mode prepends a group's training tail only when that group's rows all come after training's last row
     by the order column (`history_from_training`); otherwise the file carries its own history or is refused.
   - Predict mode scores unlabelled rows (`get_data(..., keep_unlabeled=True)`; their target stays missing).
   - Predict mode refuses a run's `input/test` copy (training's file after transforms and drops) with a message to
     put the raw rows in `data/test`, unless that copy has every raw column (alphapy-sports writes its raw
     prediction frame there).
   - Predict mode refuses to run with `live_results` set. alphapy-sports' `update_live_results`
     (`sflow/common/data_operations.py`) reads `most_recent_file(output_dir, "ranked_test*.csv")` after the
     pipeline, so a predict run would publish training's test predictions as live. Writing predictions under a
     `ranked_test*` name was rejected: the newest-file rule goes by creation time, which a copied or synced run
     directory scrambles. To use predict mode with live results, alphapy-sports reads
     `"ranked_predict*.csv" if model_specs["predict_mode"] else "ranked_test*.csv"` there and passes
     `live_results=False` to alphapy-pro's model in predict mode (its own flag still drives publishing).
   - Estimators load on first use; training no longer stacks the tabular matrices when the sequence section is
     off; the training tail selects the window columns before converting to pandas.
   - A fractional value in a column training read as integers is refused (it used to be truncated).
   - Predict mode writes its ranked file through `save_predictions` (new `output_file` argument).
7. **Close out.**
   - Full alphapy-pro suite.
   - alphapy-finance `tests/systems` and ML tests with a thread timeout (it imports this checkout editable).
   - Restart its server, Fable review, pull request to `main`.
   - Rerunning the crypto cell after step 4 measures the look-ahead. That rerun scores held-back rows with a
     changed model, so its trial line is written in the Record before it runs.
   Status: pending.

## Follow-ups (from the review, 2026-11-15; not fixed here)

- The best model is chosen on test labels when they exist (`alphapy_main.py` generate_results on the test
  partition, `select_best_model` in `model.py`), so "best" reads the held-back rows.
- LOFO (`features.py`, `KFold`), RFECV (`optimize.py`) and Optuna grid search (`optimize.py`) folds ignore
  `cv_time_blocks`, so they can train on later rows than they test.
- User transforms run on training and test rows together in training and on the frame alone at inference; a
  transform that reads across rows sees different rows.
- The regression blend is rounded to integers (`predict_blend` in `model.py`, copied in `alphapy/inference.py` so
  inference matches training).
- The window history check cannot see a gap: a frame whose first row comes after training's last row by the order
  column gets the training tail even if bars are missing in between.
- A text column given dates as datetimes at inference but read as text in training gets different text codes
  (cast to text differs from the CSV's text).
- Count features: `nan_count` counts nulls only; a float NaN (not null) is counted as present.
- `log_transform` on with an all-missing training column: the median imputer drops the column, and the
  log-transform test on the empty array is not guarded.

## Effect on alphapy-finance

- The next-bar study's AlphaPy runs fitted scaling and imputation on development and held-back rows together. That
  is a look-ahead, to be recorded in its pre-registration Record.
- Rerunning a cell after step 2 shows its size.
- The paper-study plan (alphapy-finance `docs/plans/2026-11-15-004-feat-next-bar-crypto-paper-study-plan.md`,
  step 3) uses `alphapy.inference` instead of its own copy of the feature steps.
