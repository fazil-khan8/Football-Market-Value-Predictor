"""
Day 6 - Exploratory Data Analysis.

Investigates:
- Age vs market value
- Goals/assists vs market value
- League vs market value
- Position vs market value

Saves charts as PNGs into reports/figures/ and prints summary stats.

Run from the project root:
    python3 src/data/eda.py
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

PROCESSED = "data/processed"
FIG_DIR = "reports/figures"

df = pd.read_csv(f"{PROCESSED}/player_seasons.csv")
print(f"Loaded {len(df):,} player-season rows")

# Use log market value for plotting since it's heavily skewed
df["log_market_value"] = np.log10(df["market_value_eur"])

plt.style.use("seaborn-v0_8-whitegrid")

# ---------------------------------------------------------------
# 1. Age vs market value
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
age_bucket = df.groupby(df["age"].round())["market_value_eur"].median()
ax.plot(age_bucket.index, age_bucket.values / 1_000_000, marker="o")
ax.set_xlabel("Age")
ax.set_ylabel("Median Market Value (€M)")
ax.set_title("Median Market Value by Age")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/age_vs_value.png", dpi=120)
plt.close()
print("Saved age_vs_value.png")
print(f"  Peak age (highest median value): {age_bucket.idxmax():.0f} "
      f"(€{age_bucket.max()/1_000_000:.2f}M)")

# ---------------------------------------------------------------
# 2. Goals vs market value (scatter, log value)
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df["goals"], df["log_market_value"], alpha=0.15, s=10)
ax.set_xlabel("Goals (season)")
ax.set_ylabel("log10(Market Value €)")
ax.set_title("Goals vs Market Value")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/goals_vs_value.png", dpi=120)
plt.close()
print("Saved goals_vs_value.png")
print(f"  Correlation (goals, log market value): {df['goals'].corr(df['log_market_value']):.3f}")

# ---------------------------------------------------------------
# 3. Assists vs market value
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(df["assists"], df["log_market_value"], alpha=0.15, s=10, color="orange")
ax.set_xlabel("Assists (season)")
ax.set_ylabel("log10(Market Value €)")
ax.set_title("Assists vs Market Value")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/assists_vs_value.png", dpi=120)
plt.close()
print("Saved assists_vs_value.png")
print(f"  Correlation (assists, log market value): {df['assists'].corr(df['log_market_value']):.3f}")

# ---------------------------------------------------------------
# 4. League vs market value
# ---------------------------------------------------------------
league_names = {
    "GB1": "Premier League", "ES1": "La Liga", "L1": "Bundesliga",
    "IT1": "Serie A", "FR1": "Ligue 1"
}
df["league"] = df["competition_id"].map(league_names)

fig, ax = plt.subplots(figsize=(8, 5))
league_order = df.groupby("league")["market_value_eur"].median().sort_values(ascending=False)
ax.bar(league_order.index, league_order.values / 1_000_000)
ax.set_ylabel("Median Market Value (€M)")
ax.set_title("Median Market Value by League")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/league_vs_value.png", dpi=120)
plt.close()
print("Saved league_vs_value.png")
print(league_order.apply(lambda x: f"€{x/1_000_000:.2f}M"))

# ---------------------------------------------------------------
# 5. Position vs market value
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
pos_order = df.groupby("position")["market_value_eur"].median().sort_values(ascending=False)
ax.bar(pos_order.index, pos_order.values / 1_000_000, color="green")
ax.set_ylabel("Median Market Value (€M)")
ax.set_title("Median Market Value by Position")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/position_vs_value.png", dpi=120)
plt.close()
print("Saved position_vs_value.png")
print(pos_order.apply(lambda x: f"€{x/1_000_000:.2f}M"))

# ---------------------------------------------------------------
# 6. Overall market value distribution (log scale)
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(df["log_market_value"], bins=40, color="purple", alpha=0.7)
ax.set_xlabel("log10(Market Value €)")
ax.set_ylabel("Count")
ax.set_title("Market Value Distribution (log scale)")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/value_distribution.png", dpi=120)
plt.close()
print("Saved value_distribution.png")

print(f"\nAll charts saved to {FIG_DIR}/")