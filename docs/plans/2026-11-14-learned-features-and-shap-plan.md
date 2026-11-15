# Learned features, sequence models and SHAP in AlphaPy (2026-11-14)

Goal: AlphaPy models that build their own features from raw bars, alongside the tabular models, plus SHAP explanations
for every model. The first consumer is the alphapy-finance next-bar study, which looks for the conditions that raise a
calibrated probability that the next close is higher. Names stay generic (group, order, channels), because
alphapy-sports installs this library too.

## Techniques

| Technique | Kind | Implementation | Licence |
|---|---|---|---|
| SHAP | explanation | `shap` (TreeExplainer, LinearExplainer, a permutation explainer otherwise) | MIT |
| MiniRocket / ROCKET | feature learner on windows | `aeon` or `sktime` (the original repository is GPL; not used) | BSD-3 |
| tsfresh | feature generation and FRESH selection on windows | `tsfresh` | MIT |
| Genetic programming | formula learner on windows or tabular columns | `gplearn` `SymbolicTransformer` | BSD-3 |
| 1D CNN, LSTM / GRU, InceptionTime, TCN, patch transformer | end-to-end classifiers on windows | torch (`tsai` only if its install moves nothing) | Apache-2.0 / BSD |
| CNN + LSTM over bars | end-to-end classifier | torch. DeepLOB itself needs order-book data, which the bar projects lack, so this is a different model and named as such | — |
| Chart-image CNN (Jiang, Kelly and Xiu, Journal of Finance 2023) | end-to-end classifier on rendered price-volume images | torch, from the paper | — |
| TS2Vec | self-supervised representation learner on windows | torch; MIT code reimplemented or vendored with attribution | MIT |
| Kronos | pretrained candlestick model, frozen, embeddings as features | `shiyu-coder/Kronos`. Check the weights' licence on the model card before use | MIT (repository) |
| RL formula search (AlphaGen, KDD 2023) | formula generator | reimplemented from the paper. The repository has no licence, so no code is copied | — |

## Contracts

**1. Fitting inside each fold.** AlphaPy's feature stage (`create_features(model, X_all, ...)`) runs once on the
training and test rows together, and scales them jointly. A learner fitted there would see the test rows, and every
fold score and calibration fit would then use features fitted to that fold's test rows. tsfresh selection, genetic
programming and formula search also read the labels.

So every learner is an sklearn transformer in a `Pipeline` with its classifier, registered through
`estimator_map`'s optional-package pattern and configured in `algos.yml`. Cross-validation, calibration, LOFO and
the time-ordered folds then refit it per fold automatically. Each learner gets a test that its `fit` receives only
the fold's training rows.

**2. Raw windows.** A new `model.yml` section:

```yaml
sequence:
  option: true
  group: ticker          # windows never cross groups
  order: date            # sort key within a group
  channels: [open, high, low, close, volume]
  window: 20             # bars, the row's own bar last
  session: null          # optional column; when set, a window does not cross a change in it
```

- A row's window holds only its own group's bars up to and including its own bar, all known at its close. Rows
  interleaved across groups are handled by sorting on (group, order) and mapping back.
- Short history is padded with NaN and given a mask channel. No row is dropped, because the time-ordered folds need
  one block value per training row.
- Each window is normalised within itself (prices over the window's last close, volume over the window's mean), so
  no statistic crosses rows.
- Windows are built over the training and test rows together, so a test row's window includes the bars before it.
  That uses only past bars.
- The window block joins `X` after `create_features`, so it skips the tabular median imputation and joint scaling.
  Each algorithm names its input in `algos.yml`: `input: tabular | window | both`. A column selector at the head of
  its pipeline takes that block.
- Memory: rows × channels × window float32. Liquid daily stocks at 4.4M rows × 5 × 20 is 1.8 GB. The builder logs
  the size and refuses when it exceeds a configured limit.

Every window technique gets two tests: it finds a planted pattern visible only in the window's shape, and no row's
window or features change when later bars change.

**3. Dependencies.** The alphapy-finance venv is shared with the running mflow server and vectorbt.pro.
- Dry-run each install first (`uv pip install --dry-run`) and read what it would change.
- If a library would move numpy, numba, torch or scikit-learn, it is not installed; the network is written directly
  in torch instead.
- New libraries go in optional extras only (`[sequence]`, `[explain]`), with `uv.lock` updated.

**4. Hardware.** M4 Max with 48 GB:
- estimate peak memory before every fit;
- torch uses MPS;
- one heavy fit at a time, never beside another AlphaPy run;
- see `~/.claude/CLAUDE.md` "Hardware limits".

## Framework requirement (2026-11-14)

Every new model fits AlphaPy's existing framework and configuration:
- registered in `estimator_map` as optional packages and configured in `config/algos.yml`, with AlphaPy's own
  `seed` and `n_jobs` substituted in;
- `sequence` is a `model.yml` section;
- AlphaPy passes the window shape to window algorithms;
- `apply_feature_selection_support` is the single gate: tabular algorithms never see the window block, and window
  algorithms get their tabular columns plus the block;
- univariate selection, LOFO and RFE read tabular columns only;
- calibration, the time-ordered folds, the blend and SHAP work unchanged;
- an end-to-end pipeline test runs tabular and window algorithms side by side.

## Phases

- [x] P1 (7bb7293, review fixes after Fable's review) SHAP for the existing estimators: a seeded sample of test rows; the base estimator of a calibrated model
      (values are before calibration); beeswarm and bar plots and a values CSV per algorithm; `plots: shap` in
      `model.yml`. Tests.
- [x] P2 (dbaf111 builder; pipeline integration in the framework commit) The `sequence` section and window builder, the window block in `X`, and `input` selection in `algos.yml`.
      Tests: planted pattern, no look-ahead, short history, group interleaving, session rule.
- [x] P3 (dbaf111) Fold-fitted learners: MiniRocket, tsfresh (extraction and FRESH selection inside the fold), `gplearn`
      `SymbolicTransformer`. Tests.
- [x] P4 torch classifiers (written in torch directly; channels standardised on the fitted rows; early stopping only
      after `min_epochs`, because an LSTM stopped on its initial flat loss and kept untrained weights; torch held to one
      CPU thread, because LightGBM's and torch's OpenMP runtimes deadlocked an LSTM in one process) with an sklearn interface: 1D CNN, LSTM/GRU, InceptionTime, TCN, patch transformer, CNN +
      LSTM over bars. Early stopping on a time-ordered internal split; fixed seeds. Tests.
- [ ] P5 Chart-image CNN after Jiang, Kelly and Xiu (images rendered from the window block). Tests.
- [ ] P6 TS2Vec representation learner. Tests.
- [ ] P7 Kronos embeddings (frozen), after the weights' licence check. Tests.
- [ ] P8 RL formula search reimplemented from the AlphaGen paper. Tests.
- [ ] P9 SHAP for window models: attributions by bar position and channel (a gradient explainer for torch, grouped
      permutation otherwise), not by thousands of kernel outputs.

Each phase:
- runs the full `tests/` suite and updates the `model.yml` template and README;
- gets a Fable review of the full diff before push (the global orchestration rule: Opus writes, Fable reviews);
- is committed to `develop`, then a pull request `develop` → `main` (`CLAUDE.local.md`).

Trying these models on the next-bar cells is separate work in alphapy-finance. Each model is a counted trial there,
chosen on development folds, because the held-back period has already been read many times.
