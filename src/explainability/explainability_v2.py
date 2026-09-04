"""
Day 12 v2 - SHAP explainability on the enhanced (defensive/keeper
stats) model. This is the definitive check: do defenders and
goalkeepers now get valued on defense/keeper stats specifically,
instead of goals?

Run from the project root:
    python3 src/explainability/explainability_v2.py
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import json
import shap
import matplotlib.pyplot as plt

PROCESSED = "data/processed"
MODELS = "models"
FIG_DIR = "reports/figures"

df = pd.read_csv(f"{PROCESSED}/player_seasons_features_v2.csv")

drop_cols = ["player_id", "season", "name", "market_value_eur", "log_market_value_eur"]
X = df.drop(columns=drop_cols)
y = df["log_market_value_eur"]
X["height_in_cm"] = X["height_in_cm"].fillna(X["height_in_cm"].median())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with open(f"{MODELS}/best_model_v2_info.json") as f:
    info = json.load(f)
print(f"Explaining: {info['name']}")

model = joblib.load(f"{MODELS}/best_model_v2.pkl")

if info.get("needs_scaling"):
    raise SystemExit("Best v2 model is Linear Regression -- TreeExplainer needs a tree model.")

explainer = shap.TreeExplainer(model)

sample_size = min(2000, len(X_test))
X_sample = X_test.sample(sample_size, random_state=42)
shap_values = explainer.shap_values(X_sample)
shap_df = pd.DataFrame(shap_values, columns=X_sample.columns, index=X_sample.index)

global_importance = shap_df.abs().mean().sort_values(ascending=False)
print("\n" + "=" * 70)
print("GLOBAL FEATURE IMPORTANCE (v2, mean |SHAP value|)")
print("=" * 70)
print(global_importance.head(20))

shap.summary_plot(shap_values, X_sample, show=False)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/shap_summary_v2.png", dpi=120)
plt.close()
print(f"\nSaved shap_summary_v2.png to {FIG_DIR}/")

position_cols = [c for c in X_sample.columns if c.startswith("position_")]
positions = X_sample[position_cols].idxmax(axis=1).str.replace("position_", "")

stat_features = [
    "goals", "assists", "goals_per_90", "assists_per_90",
    "goal_contributions_per_90", "minutes_played", "appearances_count",
    "age", "seasons_of_experience", "height_in_cm",
    "tackles", "tackles_won", "interceptions", "interceptions_per_90",
    "clearances", "clearances_per_90", "blocks", "defensive_actions_per_90",
    "goals_against", "saves", "save_pct", "clean_sheets", "has_advanced_stats",
]

print("\n" + "=" * 70)
print("TOP FEATURES BY POSITION (v2, includes defensive/keeper stats)")
print("=" * 70)
for pos in positions.unique():
    mask = positions == pos
    pos_importance = shap_df.loc[mask, stat_features].abs().mean().sort_values(ascending=False)
    print(f"\n{pos} (n={mask.sum()}):")
    print(pos_importance.head(8).to_string())

print(f"\nDone. Compare this against the v1 explainability output --")
print(f"specifically check whether tackles/interceptions now show up for")
print(f"Defender/Midfield, and saves/clean_sheets/goals_against for Goalkeeper.")