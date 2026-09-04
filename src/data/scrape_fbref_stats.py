"""
Extension - Scrape defensive & goalkeeper stats from FBref.

This is a pilot run: just the last 3 seasons, to confirm the pipeline
works before committing to scraping all the way back to 2012 (which
would take much longer and risks FBref rate-limiting us).

Pulls two stat types per player-season:
- "defense": tackles, interceptions, clearances, blocks (for outfield
  players -- this is what we're missing for defenders/midfielders)
- "keeper": saves, goals against, save % (what we're missing for
  goalkeepers)

Run from the project root:
    python3 src/data/scrape_fbref_stats.py
"""
import soccerdata as sd
import pandas as pd

RAW = "data/raw"

LEAGUES = "Big 5 European Leagues Combined"
SEASONS = ["2023-24", "2024-25", "2025-26"]

print(f"Fetching FBref data for {LEAGUES}, seasons: {SEASONS}")
print("(This downloads and caches data locally -- first run is slower,")
print(" re-runs will be fast since soccerdata caches to ~/soccerdata/data/FBref)")

fbref = sd.FBref(leagues=LEAGUES, seasons=SEASONS)

def flatten_columns(df):
    """FBref returns multi-level column headers (e.g. ('Tackles','Tkl')).
    Flatten them into single strings so they're easy to work with."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(c) for c in col if c and "Unnamed" not in str(c)])
                       for col in df.columns]
    return df

print("\nFetching defensive stats...")
defense = fbref.read_player_season_stats(stat_type="defense")
defense = flatten_columns(defense.reset_index())
defense.to_csv(f"{RAW}/fbref_defense_pilot.csv", index=False)
print(f"Got {len(defense):,} rows")
print("Columns:", list(defense.columns))
print(defense.head(3).to_string())

print("\nFetching goalkeeper stats...")
keeper = fbref.read_player_season_stats(stat_type="keeper")
keeper = flatten_columns(keeper.reset_index())
keeper.to_csv(f"{RAW}/fbref_keeper_pilot.csv", index=False)
print(f"Got {len(keeper):,} rows")
print("Columns:", list(keeper.columns))
print(keeper.head(3).to_string())

print(f"\nSaved to {RAW}/fbref_defense_pilot.csv and {RAW}/fbref_keeper_pilot.csv")
print("Paste the column lists and sample rows back so we can build the merge step.")