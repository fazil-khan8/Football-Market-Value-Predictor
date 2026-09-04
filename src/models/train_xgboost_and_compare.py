"""
Day 10 - XGBoost + full model comparison.

Trains XGBoost, then loads the Day 8 (Linear Regression) and Day 9
(Random Forest) models to compare all three side by side on the exact
same test set. Saves whichever model wins as models/best_model.pkl,
so Days 11-15 can just load "the best model" without caring which
algorithm it is.

Run from the project root:
    python3 src/models/train_xgboost_and_compare.py
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import json

try:
    from xgboost import XGBRegressor
except ImportError:
    raise ImportError("Run: pip3 install xgboost")

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

xgb_model = XGBRegressor(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)
joblib.dump(xgb_model, f"{MODELS}/xgboost.pkl")

def evaluate(model, X_test, y_test, needs_scaling=False, scaler=None):
    X_eval = scaler.transform(X_test) if needs_scaling else X_test
    y_pred_log = model.predict(X_eval)
    mae_log = mean_absolute_error(y_test, y_pred_log)
    rmse_log = np.sqrt(mean_squared_error(y_test, y_pred_log))
    r2_log = r2_score(y_test, y_pred_log)

    y_test_eur = np.expm1(y_test)
    y_pred_eur = np.expm1(y_pred_log)
    mae_eur = mean_absolute_error(y_test_eur, y_pred_eur)
    rmse_eur = np.sqrt(mean_squared_error(y_test_eur, y_pred_eur))
    r2_eur = r2_score(y_test_eur, y_pred_eur)

    return {
        "mae_log": mae_log, "rmse_log": rmse_log, "r2_log": r2_log,
        "mae_eur": mae_eur, "rmse_eur": rmse_eur, "r2_eur": r2_eur,
    }

results = {}

try:
    lin_model = joblib.load(f"{MODELS}/baseline_linear_regression.pkl")
    scaler = joblib.load(f"{MODELS}/baseline_scaler.pkl")
    results["Linear Regression"] = evaluate(lin_model, X_test, y_test, needs_scaling=True, scaler=scaler)
except FileNotFoundError:
    print("Warning: baseline model not found, skipping. Run Day 8 script first.")

try:
    rf_model = joblib.load(f"{MODELS}/random_forest.pkl")
    results["Random Forest"] = evaluate(rf_model, X_test, y_test)
except FileNotFoundError:
    print("Warning: random forest model not found, skipping. Run Day 9 script first.")

results["XGBoost"] = evaluate(xgb_model, X_test, y_test)

comparison_df = pd.DataFrame(results).T
comparison_df = comparison_df[["r2_log", "mae_log", "rmse_log", "r2_eur", "mae_eur", "rmse_eur"]]
comparison_df.columns = ["R2 (log)", "MAE (log)", "RMSE (log)", "R2 (eur)", "MAE (eur)", "RMSE (eur)"]

print("=" * 90)
print("MODEL COMPARISON (all evaluated on the identical held-out test set)")
print("=" * 90)
with pd.option_context("display.float_format", lambda x: f"{x:,.3f}" if abs(x) < 1000 else f"{x:,.0f}"):
    print(comparison_df)

best_name = comparison_df["R2 (log)"].astype(float).idxmax()
print(f"\nBest model (highest log-scale R2): {best_name}")

model_lookup = {"Linear Regression": lin_model if "Linear Regression" in results else None,
                "Random Forest": rf_model if "Random Forest" in results else None,
                "XGBoost": xgb_model}
best_model = model_lookup[best_name]
joblib.dump(best_model, f"{MODELS}/best_model.pkl")

with open(f"{MODELS}/best_model_info.json", "w") as f:
    json.dump({"name": best_name, "needs_scaling": best_name == "Linear Regression"}, f)

print(f"Saved winning model to {MODELS}/best_model.pkl")
print(f"Saved model metadata to {MODELS}/best_model_info.json")

importances = pd.Series(xgb_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nXGBoost top 15 feature importances:")
print(importances.head(15))