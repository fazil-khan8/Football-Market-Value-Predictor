"""
Day 11 v2 - Error analysis on the enhanced (defensive/keeper stats)
model. Same checks as the original Day 11, but this is the real test
of whether adding tackles/interceptions/saves actually helped
defenders and goalkeepers specifically, not just the aggregate score.

Run from the project root:
    python3 src/models/error_analysis_v2.py
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import json

PROCESSED = "data/processed"
MODELS = "models"

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
print(f"Analyzing errors for: {info['name']}")

model = joblib.load(f"{MODELS}/best_model_v2.pkl")

if info.get("needs_scaling"):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(X_train)
    y_pred_log = model.predict(scaler.transform(X_test))
else:
    y_pred_log = model.predict(X_test)

results = pd.DataFrame({
    "actual_eur": np.expm1(y_test.values),
    "predicted_eur": np.expm1(y_pred_log),
}, index=X_test.index)

results["abs_error_eur"] = (results["actual_eur"] - results["predicted_eur"]).abs()
results["pct_error"] = results["abs_error_eur"] / results["actual_eur"] * 100
results["signed_error_eur"] = results["predicted_eur"] - results["actual_eur"]

position_cols = [c for c in X_test.columns if c.startswith("position_")]
results["position"] = X_test[position_cols].idxmax(axis=1).str.replace("position_", "")
results["has_advanced_stats"] = X_test["has_advanced_stats"].values

print("\n" + "=" * 70)
print("ERROR BY POSITION (v2 model, with defensive/keeper stats)")
print("=" * 70)
by_position = results.groupby("position").agg(
    n=("actual_eur", "count"),
    mae_eur=("abs_error_eur", "mean"),
    median_pct_error=("pct_error", "median"),
    mean_signed_error_eur=("signed_error_eur", "mean"),
).round(0)
print(by_position)

print("\nFor reference, v1 median % error by position was:")
print("Attack: 45%, Defender: 50%, Goalkeeper: 47%, Midfield: 48%")

print("\n" + "=" * 70)
print("ERROR BY POSITION, SPLIT BY WHETHER ADVANCED STATS WERE MATCHED")
print("=" * 70)
by_position_advstats = results.groupby(["position", "has_advanced_stats"]).agg(
    n=("actual_eur", "count"),
    median_pct_error=("pct_error", "median"),
).round(1)
print(by_position_advstats)
print("\n(If median_pct_error is meaningfully LOWER for has_advanced_stats=1 rows,")
print(" especially for Defender/Goalkeeper, that's evidence the new stats genuinely help")
print(" beyond just being a recency signal.)")

results.to_csv(f"{PROCESSED}/error_analysis_results_v2.csv", index=False)
print(f"\nSaved to {PROCESSED}/error_analysis_results_v2.csv")