"""
Day 12 - Explainability (SHAP).

This is where we get a real answer to the position-fairness question:
for each position group, which features actually drive the model's
predictions? If defenders are being valued mostly on goals (like an
attacker would be), that's a real problem this will expose. If
defenders are valued on different features than attackers, that's
confirmation the model learned position-appropriate logic.

Also produces a single-player breakdown in the same format as the
original app mockup: feature -> euro contribution.

Run from the project root:
    python3 src/explainability/explainability.py
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

df = pd.read_csv(f"{PROCESSED}/player_seasons_features.csv")

drop_cols = ["player_id", "season", "name", "market_value_eur", "log_market_value_eur"]
X = df.drop(columns=drop_cols)
y = df["log_market_value_eur"]
X["height_in_cm"] = X["height_in_cm"].fillna(X["height_in_cm"].median())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with open(f"{MODELS}/best_model_info.json") as f:
    info = json.load(f)
print(f"Explaining: {info['name']}")

model = joblib.load(f"{MODELS}/best_model.pkl")

if info.get("needs_scaling"):
    raise SystemExit(
        "Best model is Linear Regression, which TreeExplainer doesn't support. "
        "Use shap.LinearExplainer instead, or re-run Day 10 -- XGBoost/RF are expected to win."
    )

explainer = shap.TreeExplainer(model)

sample_size = min(2000, len(X_test))
X_sample = X_test.sample(sample_size, random_state=42)
shap_values = explainer.shap_values(X_sample)
shap_df = pd.DataFrame(shap_values, columns=X_sample.columns, index=X_sample.index)

global_importance = shap_df.abs().mean().sort_values(ascending=False)
print("\n" + "=" * 70)
print("GLOBAL FEATURE IMPORTANCE (mean |SHAP value|, log-value scale)")
print("=" * 70)
print(global_importance.head(15))

shap.summary_plot(shap_values, X_sample, show=False)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/shap_summary.png", dpi=120)
plt.close()
print(f"\nSaved shap_summary.png to {FIG_DIR}/")

position_cols = [c for c in X_sample.columns if c.startswith("position_")]
positions = X_sample[position_cols].idxmax(axis=1).str.replace("position_", "")

stat_features = [
    "goals", "assists", "goals_per_90", "assists_per_90",
    "goal_contributions_per_90", "minutes_played", "appearances_count",
    "age", "seasons_of_experience", "height_in_cm"
]

print("\n" + "=" * 70)
print("TOP FEATURES BY POSITION (mean |SHAP value| among stat features)")
print("=" * 70)
for pos in positions.unique():
    mask = positions == pos
    pos_importance = shap_df.loc[mask, stat_features].abs().mean().sort_values(ascending=False)
    print(f"\n{pos} (n={mask.sum()}):")
    print(pos_importance.head(6).to_string())

example_idx = X_sample.index[np.expm1(y.loc[X_sample.index]).values.argmax()]
example_shap = shap_df.loc[example_idx].sort_values(key=abs, ascending=False)
base_value_log = explainer.expected_value
predicted_log = model.predict(X_sample.loc[[example_idx]])[0]

player_name = df.loc[example_idx, "name"]
actual_value = df.loc[example_idx, "market_value_eur"]

print("\n" + "=" * 70)
print(f"EXAMPLE: {player_name}")
print("=" * 70)
print(f"Actual market value: €{actual_value:,.0f}")
print(f"Predicted market value: €{np.expm1(predicted_log):,.0f}")
print(f"\nBase value (average player, log scale): {base_value_log:.3f}")
print("\nTop feature contributions (approx € impact, converting each SHAP")
print("log-contribution to a euro delta from the base prediction):")

base_eur = np.expm1(base_value_log)
for feat, shap_val in example_shap.head(8).items():
    approx_eur_impact = np.expm1(base_value_log + shap_val) - base_eur
    sign = "+" if approx_eur_impact >= 0 else ""
    print(f"  {feat:30s} {sign}€{approx_eur_impact:,.0f}")

print(f"\nDone. Full SHAP summary chart saved to {FIG_DIR}/shap_summary.png")