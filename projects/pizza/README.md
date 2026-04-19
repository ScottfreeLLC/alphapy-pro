# Pizza Toppings Trend Predictor

Learning-to-rank project that predicts which **emerging and "undiscovered"**
pizza toppings are most likely to break out, using AlphaPy's XGBoost ranker
(`XGRK`). Within each topping category (cheese / meat / sauce / produce /
herb_spice / etc.) the model ranks candidates by predicted trend momentum.

## Task framing

- **Model type:** `ranking` (pairwise)
- **Algorithm:** `XGRK` (XGBoost `rank:pairwise`)
- **Group:** `category` — toppings are ranked within their peer category so
  cheeses compete with cheeses, sauces with sauces, etc.
- **Target:** `trend_score` — 0–100 momentum score blending 12-month growth
  and chef adoption. Values for classic and already-trending toppings were
  anchored to real 2023–2026 trend data (see Sources). Test rows are blank
  — the model predicts them.

## Data

| File | Rows | Purpose |
|------|-----:|---------|
| `data/train.csv` | 86 | Classic, rising, and currently-trending toppings with observed `trend_score` |
| `data/test.csv`  | 56 | Emerging candidates and speculative "undiscovered" combinations — `trend_score` blank |

### Features (23)

- **Flavor profile (0–10):** `sweet`, `salt`, `umami`, `heat`, `acid`, `bitter`, `fat`
- **Texture / behavior (0–10):** `crunch`, `melt`, `moist`, `heat_tol`
- **Market signals (0–10):** `buzz` (social), `chef_adopt`, `novelty`, `insta`
- **Context:** `years_on_menu`, `price_tier` (1–5), `is_fermented`, `is_plant_based`, `is_premium`
- **Categoricals:** `cuisine`, `flavor_family` (target-encoded via `factors`)
- **Grouping:** `category`

`name` is carried for labeling but dropped from training via the `drop` list.

### Data sources anchoring the training labels

- PMQ Pizza Power Report 2026 (plant-based pepperoni, paneer, Indian fusion rising)
- PMQ Pizza Power Report 2025 (hot honey +430%)
- Tastewise 2026 pizza topping trends
- Datassential / Toast / Pizza Today 2025–2026 coverage

Emerging-candidate feature values are reasoned from flavor-science analogues
and trend proximity to labeled toppings, not fabricated observations.

## Regenerate the dataset

```
python build_dataset.py
```

Rows are sorted by `category` then `name` so XGBoost ranker's contiguous-group
requirement is satisfied.

## Run

From this directory:

```
alphapy
```

Outputs land in `runs/run_<timestamp>/`:

- `output/ranked_train.csv` — fitted scores for training toppings
- `output/ranked_test.csv`  — predicted scores for emerging toppings (`pred_test_xgrk`)
- `model/model.pkl`         — serialized best model
- `plots/feature_importance_train_XGRK.png`

To re-score after retraining the same saved model:

```
alphapy --predict --rundir run_<timestamp>
```

## Config highlights (`config/model.yml`)

- `live_results: True` — keeps test rows whose `trend_score` is null (pure inference)
- `ranking.group_id: category`
- `scoring_function: dcg_score`
- `factors: [cuisine, flavor_family]` — target-encoded
- `allow_na_targets: True`
- `estimators: 501`, `cv_folds: 3`, `split: 0.3`

## Headline result (latest run)

Top-ranked emerging candidate overall: **Paneer Makhani** (Indian butter-chicken
lineage), followed by **Gochujang Honey Glaze**, **Ssamjang**, **Miso Butter**,
and **Matcha Cream**. Within-category leaders include **Koji-Aged Bacon**
(meat), **Bonito Flakes** (seafood), **Black Garlic Confit** (produce), and
**Confit Egg Yolk** (protein). The model strongly favors Asian-fusion, Indian,
and umami-forward candidates — consistent with PMQ's 2026 trend thesis.
