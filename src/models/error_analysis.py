"""
Day 11 - Error analysis.

Loads the winning model (models/best_model.pkl, picked on Day 10) and
digs into WHERE it's wrong:
- Error by position (the key thing we're tracking: are defenders/
  goalkeepers/midfielders predicted as reliably as attackers?)
- Error by league
- Error by value range (are expensive players harder to predict?)
- Whether the model is systematically biased toward over- or
  under-predicting in certain ranges

Run from the project root:
    python3 src/models/error_analysis.py
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import json

PROCESSED = "data/processed"
MODELS = "models"

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
print(f"Analyzing errors for: {info['name']}")

model = joblib.load(f"{MODELS}/best_model.pkl")

if info.get("needs_scaling"):
    scaler = joblib.load(f"{MODELS}/baseline_scaler.pkl")
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

league_cols = [c for c in X_test.columns if c.startswith("league_")]
results["league"] = X_test[league_cols].idxmax(axis=1).str.replace("league_", "")

print("\n" + "=" * 70)
print("ERROR BY POSITION")
print("=" * 70)
by_position = results.groupby("position").agg(
    n=("actual_eur", "count"),
    median_actual_eur=("actual_eur", "median"),
    mae_eur=("abs_error_eur", "mean"),
    median_pct_error=("pct_error", "median"),
    mean_signed_error_eur=("signed_error_eur", "mean"),
).round(0)
print(by_position)
print("\n(mean_signed_error_eur > 0 means the model OVER-values that position on average; < 0 means it UNDER-values them)")

print("\n" + "=" * 70)
print("ERROR BY LEAGUE")
print("=" * 70)
by_league = results.groupby("league").agg(
    n=("actual_eur", "count"),
    mae_eur=("abs_error_eur", "mean"),
    median_pct_error=("pct_error", "median"),
).round(0)
print(by_league)

print("\n" + "=" * 70)
print("ERROR BY VALUE RANGE")
print("=" * 70)
bins = [0, 500_000, 1_000_000, 5_000_000, 20_000_000, np.inf]
labels = ["<€500K", "€500K-1M", "€1M-5M", "€5M-20M", ">€20M"]
results["value_bucket"] = pd.cut(results["actual_eur"], bins=bins, labels=labels)
by_value = results.groupby("value_bucket", observed=True).agg(
    n=("actual_eur", "count"),
    mae_eur=("abs_error_eur", "mean"),
    median_pct_error=("pct_error", "median"),
).round(0)
print(by_value)

corr = results["actual_eur"].corr(results["abs_error_eur"])
print(f"\nCorrelation between actual value and absolute error: {corr:.3f}")
print("(positive = yes, expensive players have bigger absolute errors -- expected, since")
print(" a 10% miss on a €50M player is a much bigger euro number than a 10% miss on a €500K player)")

print("\n" + "=" * 70)
print("10 WORST ABSOLUTE MISSES")
print("=" * 70)
worst = results.merge(df[["name"]], left_index=True, right_index=True)
print(worst.sort_values("abs_error_eur", ascending=False)
      [["name", "position", "league", "actual_eur", "predicted_eur", "abs_error_eur"]]
      .head(10).to_string())

results.to_csv(f"{PROCESSED}/error_analysis_results.csv", index=False)
print(f"\nSaved full results to {PROCESSED}/error_analysis_results.csv")