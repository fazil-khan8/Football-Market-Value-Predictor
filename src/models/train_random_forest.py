"""
Day 9 - Random Forest model.

Trains a Random Forest on the same features/target as the Day 8
baseline, so we can compare them fairly. Also prints feature
importances -- an early look at what's driving predictions, and a
first check on whether the model is picking up position-specific
signal (relevant to the position-fairness concern we're tracking).

Run from the project root:
    python3 src/models/train_random_forest.py
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

PROCESSED = "data/processed"
MODELS = "models"

df = pd.read_csv(f"{PROCESSED}/player_seasons_features.csv")
print(f"Loaded {len(df):,} rows")

drop_cols = ["player_id", "season", "name", "market_value_eur", "log_market_value_eur"]
X = df.drop(columns=drop_cols)
y = df["log_market_value_eur"]
X["height_in_cm"] = X["height_in_cm"].fillna(X["height_in_cm"].median())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42
)
model.fit(X_train, y_train)

y_pred_log = model.predict(X_test)

mae_log = mean_absolute_error(y_test, y_pred_log)
rmse_log = np.sqrt(mean_squared_error(y_test, y_pred_log))
r2_log = r2_score(y_test, y_pred_log)

y_test_eur = np.expm1(y_test)
y_pred_eur = np.expm1(y_pred_log)
mae_eur = mean_absolute_error(y_test_eur, y_pred_eur)
rmse_eur = np.sqrt(mean_squared_error(y_test_eur, y_pred_eur))
r2_eur = r2_score(y_test_eur, y_pred_eur)

print("\n" + "=" * 50)
print("RANDOM FOREST")
print("=" * 50)
print("On log scale:")
print(f"  MAE:  {mae_log:.3f}")
print(f"  RMSE: {rmse_log:.3f}")
print(f"  R2:   {r2_log:.3f}")
print("\nOn real euro scale:")
print(f"  MAE:  €{mae_eur:,.0f}")
print(f"  RMSE: €{rmse_eur:,.0f}")
print(f"  R2:   {r2_eur:.3f}")

importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nTop 15 feature importances:")
print(importances.head(15))

sample_idx = X_test.index[:8]
comparison = pd.DataFrame({
    "actual_eur": np.expm1(y_test.loc[sample_idx]),
    "predicted_eur": np.expm1(model.predict(X_test.loc[sample_idx]))
})
print("\nSample predictions (same rows as Day 8 baseline):")
print(comparison.to_string())

joblib.dump(model, f"{MODELS}/random_forest.pkl")
print(f"\nSaved model to {MODELS}/random_forest.pkl")