# ⚽ AI Football Player Market Value Predictor

Predicts a football player's market value (in € million) from playing statistics,
using data from Europe's top five leagues: Premier League, La Liga, Bundesliga,
Serie A, and Ligue 1.

Supports two modes:
- **Player Search** — look up a real player and see their predicted market value.
- **Custom Prediction** — enter a hypothetical player's stats and get an estimate.

> 🚧 Work in progress. See [Roadmap](#roadmap) for current status.

## Prediction target

Decided on Day 1: **`market_value_eur`** (current market value in euros, log-transformed
during training to reduce the influence of a small number of extremely high-value players).
Source: `<fill in once dataset is chosen — e.g. Transfermarkt via Kaggle export>`.

## Project structure

```
football-market-value-predictor/
├── data/
│   ├── raw/            # untouched source datasets (gitignored)
│   └── processed/      # cleaned, merged, feature-engineered data (gitignored)
├── notebooks/           # exploratory analysis, one notebook per roadmap "day"
├── src/
│   ├── data/            # loading, cleaning, merging scripts
│   ├── features/        # feature engineering (per-90 stats, encodings)
│   ├── models/           # training, evaluation, model comparison
│   ├── explainability/  # SHAP / feature importance
│   └── app/              # Streamlit application
├── models/               # serialized trained models (gitignored)
├── reports/figures/      # exported charts for README / write-up
├── tests/
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Roadmap

Progress tracker for the ~18-session build. Checked items are done.

### Phase 0 — Planning
- [x] Day 1 — Project setup & design (repo, env, folders, target, README, .gitignore)

### Phase 1 — Data
- [ ] Day 2 — Find and download datasets (all 5 leagues)
- [ ] Day 3 — Explore the raw data
- [ ] Day 4 — Clean player data
- [ ] Day 5 — Combine datasets (identity + stats + market value)

### Phase 2 — Analysis
- [ ] Day 6 — Exploratory data analysis & visualizations

### Phase 3 — Feature Engineering
- [ ] Day 7 — Per-90 features, league/position encoding

### Phase 4 — Machine Learning
- [ ] Day 8 — Baseline linear regression
- [ ] Day 9 — Random forest model
- [ ] Day 10 — Gradient boosting / XGBoost + model comparison

### Phase 5 — Model Validation
- [ ] Day 11 — Error analysis across leagues/positions/value ranges

### Phase 6 — Explainable AI
- [ ] Day 12 — Feature importance & SHAP explanations

### Phase 7 — Web Application
- [ ] Day 13 — Streamlit app structure
- [ ] Day 14 — Player search & prediction
- [ ] Day 15 — Custom player prediction mode

### Phase 8 — Polish
- [ ] Day 16 — UI improvements

### Phase 9 — Finalize
- [ ] Day 17 — Testing & deployment
- [ ] Day 18 — Final documentation & screenshots

## License

TBD
