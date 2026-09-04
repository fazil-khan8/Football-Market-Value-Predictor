"""
Extension, v3 - Train ONLY on rows with real matched defensive/keeper
data (has_advanced_stats=1), removing the zero-filled rows entirely.
This isolates whether defensive stats genuinely help, without the
dilution/noise of ~44% zero-filled placeholder rows.

Run from the project root:
    python3 src/models/train_v3_matched_only.py
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json

PROCESSED = "data/processed"
MODELS = "models"

df = pd.read_csv(f"{PROCESSED}/player_seasons_features_v2.csv")
print(f"Loaded {len(df):,} total rows")

df = df[df["has_advanced_stats"] == 1].copy()
print(f"Filtered to {len(df):,} rows with real matched defensive/keeper data")

drop_cols = ["player_id", "season", "name", "market_value_eur", "log_market_value_eur",
             "has_advanced_stats"]
X = df.drop(columns=drop_cols)
y = df["log_market_value_eur"]
X["height_in_cm"] = X["height_in_cm"].fillna(X["height_in_cm"].median())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

def evaluate(model, X_test, y_test, needs_scaling=False, scaler=None):
    X_eval = scaler.transform(X_test) if needs_scaling else X_test
    y_pred_log = model.predict(X_eval)
    r2_log = r2_score(y_test, y_pred_log)
    mae_log = mean_absolute_error(y_test, y_pred_log)
    y_test_eur = np.expm1(y_test)
    y_pred_eur = np.expm1(y_pred_log)
    r2_eur = r2_score(y_test_eur, y_pred_eur)
    mae_eur = mean_absolute_error(y_test_eur, y_pred_eur)
    return {"r2_log": r2_log, "mae_log": mae_log, "r2_eur": r2_eur, "mae_eur": mae_eur}

results = {}

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
lin_model = LinearRegression().fit(X_train_s, y_train)
results["Linear Regression (v3)"] = evaluate(lin_model, X_test, y_test, needs_scaling=True, scaler=scaler)

rf_model = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=42)
rf_model.fit(X_train, y_train)
results["Random Forest (v3)"] = evaluate(rf_model, X_test, y_test)

xgb_model = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
xgb_model.fit(X_train, y_train)
results["XGBoost (v3)"] = evaluate(xgb_model, X_test, y_test)

comparison = pd.DataFrame(results).T
comparison.columns = ["R2 (log)", "MAE (log)", "R2 (eur)", "MAE (eur)"]
print("\n" + "=" * 70)
print("V3 MODEL COMPARISON (matched-only, no zero-filled rows)")
print("=" * 70)
with pd.option_context("display.float_format", lambda x: f"{x:,.3f}" if abs(x) < 1000 else f"{x:,.0f}"):
    print(comparison)

print("\nFor reference:")
print("v1 XGBoost (no defensive stats):        R2(log)=0.689, MAE(eur)=€5.11M")
print("v2 XGBoost (defensive stats, zero-fill): R2(log)=0.715, MAE(eur)=€4.83M")

best_name = comparison["R2 (log)"].astype(float).idxmax()
print(f"\nBest v3 model: {best_name}")
model_lookup = {"Linear Regression (v3)": lin_model, "Random Forest (v3)": rf_model, "XGBoost (v3)": xgb_model}
best_model = model_lookup[best_name]
joblib.dump(best_model, f"{MODELS}/best_model_v3.pkl")
with open(f"{MODELS}/best_model_v3_info.json", "w") as f:
    json.dump({"name": best_name, "needs_scaling": "Linear" in best_name}, f)
print(f"Saved to {MODELS}/best_model_v3.pkl")

if "XGBoost" in best_name:
    importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTop 20 feature importances (v3, matched-only):")
    print(importances.head(20))

df.to_csv(f"{PROCESSED}/player_seasons_features_v3_matched_only.csv", index=False)
print(f"\nSaved filtered dataset to {PROCESSED}/player_seasons_features_v3_matched_only.csv")