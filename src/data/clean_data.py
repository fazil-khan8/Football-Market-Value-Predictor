"""
 Clean and merge player data into one row per player per season.

What this does:
1. Rolls up appearances.csv (one row per game) into one row per
   player per season (goals, assists, minutes summed).
2. Attaches the player's market value as of the END of that season
   (using the closest valuation on or before the season-end date).
3. Attaches static player profile info (position, foot, nationality,
   height) and computes age as of that season.
4. Drops rows we can't use for training: missing position, missing
   market value, or seasons before appearances data starts (2012).
5. Standardizes column names and saves the result to
   data/processed/player_seasons.csv

Run from the project root:
    python3 src/data/clean_data.py
"""
import pandas as pd
import numpy as np

RAW = "data/raw"
PROCESSED = "data/processed"

# ---------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------
players = pd.read_csv(f"{RAW}/players.csv")
appearances = pd.read_csv(f"{RAW}/appearances.csv", parse_dates=["date"])
valuations = pd.read_csv(f"{RAW}/player_valuations.csv", parse_dates=["date"])

print(f"Loaded: players={len(players):,}, appearances={len(appearances):,}, valuations={len(valuations):,}")

# ---------------------------------------------------------------
# 2. Assign a "season" to every appearance
#    European convention: season runs Aug -> Jul.
#    e.g. a game in Oct 2021 or Mar 2022 both belong to "season 2021".
# ---------------------------------------------------------------
appearances["season"] = np.where(
    appearances["date"].dt.month >= 7,
    appearances["date"].dt.year,
    appearances["date"].dt.year - 1
)

# Drop seasons before 2012 (data coverage starts there — see Day 3 findings)
before = len(appearances)
appearances = appearances[appearances["season"] >= 2012]
print(f"Dropped {before - len(appearances):,} appearance rows before season 2012")

# ---------------------------------------------------------------
# 3. Roll appearances up to one row per player per season
# ---------------------------------------------------------------
season_stats = appearances.groupby(["player_id", "season"]).agg(
    appearances_count=("game_id", "count"),
    goals=("goals", "sum"),
    assists=("assists", "sum"),
    minutes_played=("minutes_played", "sum"),
    yellow_cards=("yellow_cards", "sum"),
    red_cards=("red_cards", "sum"),
    # most common competition that season = the league they mainly played in
    competition_id=("competition_id", lambda x: x.mode().iloc[0] if not x.mode().empty else None),
).reset_index()

print(f"Rolled up to {len(season_stats):,} player-season rows")

# ---------------------------------------------------------------
# 4. Attach market value as of season end (closest valuation on/before
#    season-end date, per player) using merge_asof
# ---------------------------------------------------------------
season_stats["season_end_date"] = pd.to_datetime(
    (season_stats["season"] + 1).astype(str) + "-06-30"
)

valuations_sorted = valuations.sort_values("date")
season_stats_sorted = season_stats.sort_values("season_end_date")

merged = pd.merge_asof(
    season_stats_sorted,
    valuations_sorted[["player_id", "date", "market_value_in_eur"]],
    left_on="season_end_date",
    right_on="date",
    by="player_id",
    direction="backward",   # most recent valuation ON OR BEFORE season end
    tolerance=pd.Timedelta(days=400),  # don't match valuations more than ~1yr away
)
merged = merged.rename(columns={"market_value_in_eur": "market_value_eur"})
merged = merged.drop(columns=["date"])

matched = merged["market_value_eur"].notna().sum()
print(f"Matched market value for {matched:,} / {len(merged):,} player-season rows")

# ---------------------------------------------------------------
# 5. Attach static player profile info + compute age at season end
# ---------------------------------------------------------------
players["date_of_birth"] = pd.to_datetime(players["date_of_birth"], errors="coerce")

profile_cols = players[[
    "player_id", "name", "position", "sub_position", "foot",
    "height_in_cm", "country_of_citizenship", "date_of_birth"
]]

final = merged.merge(profile_cols, on="player_id", how="left")

final["age"] = (
    (final["season_end_date"] - final["date_of_birth"]).dt.days / 365.25
).round(1)

# ---------------------------------------------------------------
# 6. Clean: drop rows unusable for training
# ---------------------------------------------------------------
before = len(final)
final = final[final["position"].notna() & (final["position"] != "Missing")]
print(f"Dropped {before - len(final):,} rows with missing/unknown position")

before = len(final)
final = final[final["market_value_eur"].notna()]
print(f"Dropped {before - len(final):,} rows with no matched market value")

before = len(final)
final = final[final["age"].notna() & (final["age"] > 14) & (final["age"] < 45)]
print(f"Dropped {before - len(final):,} rows with missing/implausible age")

# Drop seasons with almost no playing time — not meaningful signal
before = len(final)
final = final[final["minutes_played"] >= 90]  # at least one full match worth
print(f"Dropped {before - len(final):,} rows with under 90 total minutes played")

# ---------------------------------------------------------------
# 7. Standardize position naming (already fairly clean, but normalize case)
# ---------------------------------------------------------------
final["position"] = final["position"].str.strip().str.title()

# ---------------------------------------------------------------
# 8. Save
# ---------------------------------------------------------------
final = final.drop(columns=["season_end_date", "date_of_birth"])
final.to_csv(f"{PROCESSED}/player_seasons.csv", index=False)

print(f"\nFinal dataset: {len(final):,} player-season rows, {final.shape[1]} columns")
print(f"Saved to {PROCESSED}/player_seasons.csv")
print("\nColumn preview:")
print(final.dtypes)
print("\nSample rows:")
print(final.head(3).to_string())