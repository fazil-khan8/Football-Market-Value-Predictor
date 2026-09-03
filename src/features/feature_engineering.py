"""
Day 7 - Feature Engineering.

Turns raw counting stats into per-90 rates, adds experience/age
features, and encodes categorical variables (league, position) so the
data is ready to feed straight into a model on Day 8.

Run from the project root:
    python3 src/features/feature_engineering.py
"""
import pandas as pd
import numpy as np

PROCESSED = "data/processed"

df = pd.read_csv(f"{PROCESSED}/player_seasons.csv")
print(f"Loaded {len(df):,} player-season rows")

# ---------------------------------------------------------------
# 1. Per-90 stats
#    Raw goals/assists are biased toward players who play more
#    minutes. Per-90 puts everyone on the same footing.
# ---------------------------------------------------------------
df["goals_per_90"] = df["goals"] / df["minutes_played"] * 90
df["assists_per_90"] = df["assists"] / df["minutes_played"] * 90
df["goal_contributions_per_90"] = df["goals_per_90"] + df["assists_per_90"]

# Minutes per appearance — proxy for "is this player a regular starter
# or a bench player getting occasional minutes"
df["minutes_per_appearance"] = df["minutes_played"] / df["appearances_count"]

# ---------------------------------------------------------------
# 2. Experience feature
#    How many seasons of data do we have for this player up to and
#    including this row? Captures "established veteran" vs "just
#    broke into the first team" separately from raw age.
# ---------------------------------------------------------------
df = df.sort_values(["player_id", "season"])
df["seasons_of_experience"] = df.groupby("player_id").cumcount() + 1

# ---------------------------------------------------------------
# 3. Age features
#    Day 6 showed the raw age-value relationship is noisy at the
#    very young end (tiny sample sizes) and peaks somewhere in the
#    early-to-mid 20s then declines. A plain linear age term can't
#    capture that curve, so we add age_squared -- this lets a linear
#    model (Day 8's baseline) fit a peak-and-decline shape instead of
#    a straight line. Tree models don't strictly need this, but it
#    doesn't hurt them either.
# ---------------------------------------------------------------
df["age_squared"] = df["age"] ** 2

# ---------------------------------------------------------------
# 4. League encoding (one-hot)
# ---------------------------------------------------------------
league_names = {
    "GB1": "Premier League", "ES1": "La Liga", "L1": "Bundesliga",
    "IT1": "Serie A", "FR1": "Ligue 1"
}
df["league"] = df["competition_id"].map(league_names)
league_dummies = pd.get_dummies(df["league"], prefix="league")

# ---------------------------------------------------------------
# 5. Position encoding (one-hot)
#    Using the 4 broad positions rather than sub_position (too many
#    sparse categories to encode reliably at this dataset size).
# ---------------------------------------------------------------
position_dummies = pd.get_dummies(df["position"], prefix="position")

# ---------------------------------------------------------------
# 6. Foot encoding (one-hot, small cardinality: left/right/both)
# ---------------------------------------------------------------
foot_dummies = pd.get_dummies(df["foot"].fillna("unknown"), prefix="foot")

# ---------------------------------------------------------------
# 7. Assemble final feature table
# ---------------------------------------------------------------
feature_cols = [
    "player_id", "season", "name",  # identifiers, kept for reference/search
    "age", "age_squared", "height_in_cm", "seasons_of_experience",
    "appearances_count", "minutes_played", "minutes_per_appearance",
    "goals", "assists", "goals_per_90", "assists_per_90",
    "goal_contributions_per_90", "yellow_cards", "red_cards",
    "market_value_eur",  # target
]

final = pd.concat(
    [df[feature_cols], league_dummies, position_dummies, foot_dummies],
    axis=1
)

# Log-transform the target (decided Day 1: market value is heavily
# right-skewed, log makes it far easier for any model to learn)
final["log_market_value_eur"] = np.log1p(final["market_value_eur"])

final.to_csv(f"{PROCESSED}/player_seasons_features.csv", index=False)

print(f"\nFinal feature table: {final.shape[0]:,} rows x {final.shape[1]} columns")
print("\nColumns:")
print(list(final.columns))
print("\nSample rows:")
print(final.head(3).to_string())
print(f"\nSaved to {PROCESSED}/player_seasons_features.csv")