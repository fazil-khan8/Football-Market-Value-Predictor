"""
Day 8 - Baseline model: Linear Regression.

Trains on the log-transformed target (log_market_value_eur), then
converts predictions back to real euros for reporting so the error
numbers are interpretable.

Run from the project root:
    python3 src/models/train_baseline.py
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

PROCESSED = "data/processed"
MODELS = "models"

df = pd.read_csv(f"{PROCESSED}/player_seasons_features.csv")
print(f"Loaded {len(df):,} rows")

# ---------------------------------------------------------------
# 1. Split features (X) from target (y)
#    Drop identifiers and the raw (non-log) market value -- we only
#    train on the log target, per the Day 1 decision.
# ---------------------------------------------------------------
drop_cols = ["player_id", "season", "name", "market_value_eur", "log_market_value_eur"]
X = df.drop(columns=drop_cols)
y = df["log_market_value_eur"]
X["height_in_cm"] = X["height_in_cm"].fillna(X["height_in_cm"].median())

print(f"Feature columns ({X.shape[1]}): {list(X.columns)}")

# ---------------------------------------------------------------
# 2. Train/test split
#    80/20, random_state fixed so results are reproducible.
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\nTrain: {len(X_train):,} rows | Test: {len(X_test):,} rows")

# ---------------------------------------------------------------
# 3. Scale features
#    Linear regression benefits from standardized inputs since raw
#    features are on very different scales (age ~20-40 vs minutes
#    ~90-3000).
# ---------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------
# 4. Train
# ---------------------------------------------------------------
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# ---------------------------------------------------------------
# 5. Evaluate -- on both the log scale (what the model actually
#    optimizes) and the real-euro scale (what actually matters to a
#    person reading the report)
# ---------------------------------------------------------------
y_pred_log = model.predict(X_test_scaled)

mae_log = mean_absolute_error(y_test, y_pred_log)
rmse_log = np.sqrt(mean_squared_error(y_test, y_pred_log))
r2_log = r2_score(y_test, y_pred_log)

y_test_eur = np.expm1(y_test)
y_pred_eur = np.expm1(y_pred_log)

mae_eur = mean_absolute_error(y_test_eur, y_pred_eur)
rmse_eur = np.sqrt(mean_squared_error(y_test_eur, y_pred_eur))
r2_eur = r2_score(y_test_eur, y_pred_eur)

print("\n" + "=" * 50)
print("BASELINE: Linear Regression")
print("=" * 50)
print("On log scale (what the model optimizes):")
print(f"  MAE:  {mae_log:.3f}")
print(f"  RMSE: {rmse_log:.3f}")
print(f"  R2:   {r2_log:.3f}")
print("\nOn real euro scale (converted back, easier to interpret):")
print(f"  MAE:  €{mae_eur:,.0f}")
print(f"  RMSE: €{rmse_eur:,.0f}")
print(f"  R2:   {r2_eur:.3f}")

# ---------------------------------------------------------------
# 6. Show a few real vs predicted examples
# ---------------------------------------------------------------
sample_idx = X_test.index[:8]
comparison = pd.DataFrame({
    "actual_eur": np.expm1(y_test.loc[sample_idx]),
    "predicted_eur": np.expm1(model.predict(scaler.transform(X_test.loc[sample_idx])))
})
print("\nSample predictions:")
print(comparison.to_string())

# ---------------------------------------------------------------
# 7. Save model + scaler for later comparison (Day 10)
# ---------------------------------------------------------------
joblib.dump(model, f"{MODELS}/baseline_linear_regression.pkl")
joblib.dump(scaler, f"{MODELS}/baseline_scaler.pkl")
print(f"\nSaved model to {MODELS}/baseline_linear_regression.pkl")