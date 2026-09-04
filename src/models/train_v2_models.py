"""
Extension, step 3 - Retrain on the enhanced feature set (v2, with
defensive/keeper stats) and compare against the original models to
see if adding this data actually improved things.

Run from the project root:
    python3 src/models/train_v2_models.py
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
print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")

drop_cols = ["player_id", "season", "name", "market_value_eur", "log_market_value_eur"]
X = df.drop(columns=drop_cols)
y = df["log_market_value_eur"]
X["height_in_cm"] = X["height_in_cm"].fillna(X["height_in_cm"].median())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")
print(f"Feature count: {X.shape[1]} (was 27 before adding defensive/keeper stats)")

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
results["Linear Regression (v2)"] = evaluate(lin_model, X_test, y_test, needs_scaling=True, scaler=scaler)

rf_model = RandomForestRegressor(n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=42)
rf_model.fit(X_train, y_train)
results["Random Forest (v2)"] = evaluate(rf_model, X_test, y_test)

xgb_model = XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
xgb_model.fit(X_train, y_train)
results["XGBoost (v2)"] = evaluate(xgb_model, X_test, y_test)

comparison = pd.DataFrame(results).T
comparison.columns = ["R2 (log)", "MAE (log)", "R2 (eur)", "MAE (eur)"]
print("\n" + "=" * 70)
print("V2 MODEL COMPARISON (with defensive/keeper stats)")
print("=" * 70)
with pd.option_context("display.float_format", lambda x: f"{x:,.3f}" if abs(x) < 1000 else f"{x:,.0f}"):
    print(comparison)

print("\nFor reference, v1 results (without defensive/keeper stats) were:")
print("Linear Regression: R2(log)=0.628, MAE(eur)=€5.61M")
print("Random Forest:     R2(log)=0.669, MAE(eur)=€5.34M")
print("XGBoost:           R2(log)=0.689, MAE(eur)=€5.11M")

best_name = comparison["R2 (log)"].astype(float).idxmax()
print(f"\nBest v2 model: {best_name}")

model_lookup = {"Linear Regression (v2)": lin_model, "Random Forest (v2)": rf_model, "XGBoost (v2)": xgb_model}
best_model = model_lookup[best_name]
joblib.dump(best_model, f"{MODELS}/best_model_v2.pkl")
with open(f"{MODELS}/best_model_v2_info.json", "w") as f:
    json.dump({"name": best_name, "needs_scaling": "Linear" in best_name}, f)
print(f"Saved to {MODELS}/best_model_v2.pkl")

if "XGBoost" in best_name:
    importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("\nTop 20 feature importances (v2 model):")
    print(importances.head(20))