# AlphaPy Configuration

Two config layers: **global** (this directory) and **per-project** (in each `projects/<name>/config/`).

## Global config (this directory)

| File | Purpose | Tracked? |
|------|---------|----------|
| `alphapy.yml` | Local paths (data dir, project root). Read at startup by `alphapy_main.get_alphapy_config()`. | gitignored |
| `alphapy.yml.template` | Template — copy to `alphapy.yml` and edit. | tracked |
| `algos.yml` | ML algorithm definitions and hyperparameter grids (sklearn, XGBoost, CatBoost, LightGBM). Used by `optimize.py`. | tracked |
| `model.yml.template` | Per-project model config template — copy to `projects/<name>/config/model.yml`. | tracked |

## Per-project config

Each project under `projects/<name>/` has `config/model.yml` defining: target column, algorithms to run, CV folds, hyperparameter optimization settings, encoding, scaling, and feature selection. See `projects/kaggle/config/model.yml` for a working example.

## Quick start

```bash
cp alphapy.yml.template alphapy.yml
# Edit alphapy.yml with your data_dir and project_root paths
```

Then create a new project:

```bash
mkdir -p projects/myproject/{config,data}
cp config/model.yml.template projects/myproject/config/model.yml
# Edit model.yml: set target column, algorithms, etc.
cd projects/myproject && alphapy
```

## Notes

- Trading/markets/finance configs (`sources.yml`, `systems.yml`, `groups.yml`, `indicators.yml`, finance-flavored `variables.yml`) moved to the private `alphapy-finance` repo in v4.0.0.
- `algos.yml` is shared and domain-agnostic. All entries are pure ML algorithms.
