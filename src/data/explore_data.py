"""
Day 3 - Explore the raw data.

Answers:
- How many players / seasons / leagues?
- How is market value distributed?
- What's missing?
- Does the appearances table need to be rolled up per player-season?

Run from the project root:
    python3 src/data/explore_data.py
"""
import pandas as pd

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

RAW = "data/raw"

players = pd.read_csv(f"{RAW}/players.csv")
appearances = pd.read_csv(f"{RAW}/appearances.csv")
valuations = pd.read_csv(f"{RAW}/player_valuations.csv")
clubs = pd.read_csv(f"{RAW}/clubs.csv")
competitions = pd.read_csv(f"{RAW}/competitions.csv")

print("=" * 70)
print("SHAPES")
print("=" * 70)
for name, df in [("players", players), ("appearances", appearances),
                  ("valuations", valuations), ("clubs", clubs)]:
    print(f"{name:12s}: {df.shape[0]:>8,} rows x {df.shape[1]} cols")

print("\n" + "=" * 70)
print("PLAYERS - columns and missing values")
print("=" * 70)
print(players.dtypes)
print("\nMissing values (top 15 by count):")
print(players.isna().sum().sort_values(ascending=False).head(15))

print("\n" + "=" * 70)
print("MARKET VALUE DISTRIBUTION (players.market_value_in_eur)")
print("=" * 70)
if "market_value_in_eur" in players.columns:
    mv = players["market_value_in_eur"].dropna()
    print(mv.describe())
    print("\nPercentiles:")
    for p in [1, 5, 25, 50, 75, 95, 99]:
        print(f"  p{p}: EUR {mv.quantile(p/100):,.0f}")
    print(f"\nPlayers with NO current market value: {players['market_value_in_eur'].isna().sum():,} "
          f"({players['market_value_in_eur'].isna().mean()*100:.1f}%)")
else:
    print("Column 'market_value_in_eur' not found — check players.csv columns above.")

print("\n" + "=" * 70)
print("PLAYER VALUATIONS TABLE (historical, time-series)")
print("=" * 70)
print(valuations.dtypes)
if "date" in valuations.columns:
    valuations["date"] = pd.to_datetime(valuations["date"], errors="coerce")
    print(f"\nDate range: {valuations['date'].min()} to {valuations['date'].max()}")
    print(f"Valuations per year:")
    print(valuations["date"].dt.year.value_counts().sort_index())

print("\n" + "=" * 70)
print("APPEARANCES TABLE (one row per player per game)")
print("=" * 70)
print(appearances.dtypes)
if "date" in appearances.columns:
    appearances["date"] = pd.to_datetime(appearances["date"], errors="coerce")
    print(f"\nDate range: {appearances['date'].min()} to {appearances['date'].max()}")
elif "player_id" in appearances.columns and "game_id" in appearances.columns:
    print(f"\nUnique players in appearances: {appearances['player_id'].nunique():,}")
    print(f"Unique games in appearances: {appearances['game_id'].nunique():,}")

print("\nAppearances per player (sanity check - confirms we need to roll this up):")
per_player = appearances.groupby("player_id").size()
print(per_player.describe())

print("\n" + "=" * 70)
print("NAME MATCHING CHECK - do player names look clean/unique?")
print("=" * 70)
dupes = players["name"].duplicated().sum() if "name" in players.columns else "n/a"
print(f"Duplicate player names in players table: {dupes}")

print("\n" + "=" * 70)
print("POSITIONS")
print("=" * 70)
if "position" in players.columns:
    print(players["position"].value_counts())

print("\n" + "=" * 70)
print("DONE. Review the output above before Day 4 (cleaning).")
print("=" * 70)