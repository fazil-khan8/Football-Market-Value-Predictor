"""
Extension, step 1 - Consolidate the two FBref dataset sources into one
clean file with defensive + goalkeeper stats, standardized columns,
covering seasons 2017-2023 (rich defense+keeper stats) and 2025
(keeper stats + partial defense: interceptions/tackles-won only).

Run from the project root:
    python3 src/data/consolidate_fbref_stats.py
"""
import pandas as pd
import glob

RAW = "data/raw/fbref"
OUT = "data/raw/fbref/fbref_consolidated.csv"

rows = []

cleaned_files = sorted(glob.glob(f"{RAW}/cleaned_*.csv"))
print(f"Found {len(cleaned_files)} 'cleaned_' season files: {[f.split('/')[-1] for f in cleaned_files]}")

for path in cleaned_files:
    df = pd.read_csv(path)
    season_start = int(df["season"].iloc[0].split("-")[0])
    standardized = pd.DataFrame({
        "player": df["player"],
        "league": df["comp"],
        "season": season_start,
        "position_fbref": df["pos"],
        "tackles": df["Tackles attempted"],
        "tackles_won": df["Tackles Won"],
        "interceptions": df["Interceptions"],
        "clearances": df["Clearances"],
        "blocks": df["Shots blocked"] + df["Passes blocked"],
        "errors": df["Errors made"],
        "goals_against": df["Goals Against"],
        "saves": df["Saves"],
        "save_pct": df["Saves %"],
        "clean_sheets": df["Clean Sheets"],
    })
    rows.append(standardized)

df2 = pd.read_csv(f"{RAW}/players_data-2025_2026.csv")
league_clean = df2["Comp"].str.split(" ", n=1).str[1]

standardized2 = pd.DataFrame({
    "player": df2["Player"],
    "league": league_clean,
    "season": 2025,
    "position_fbref": df2["Pos"],
    "tackles": pd.NA,
    "tackles_won": df2["TklW"],
    "interceptions": df2["Int"],
    "clearances": pd.NA,
    "blocks": pd.NA,
    "errors": pd.NA,
    "goals_against": df2["GA"],
    "saves": df2["Saves"],
    "save_pct": df2["Save%"],
    "clean_sheets": df2["CS"],
})
rows.append(standardized2)

consolidated = pd.concat(rows, ignore_index=True)

print(f"\nConsolidated: {len(consolidated):,} rows")
print(f"Seasons covered: {sorted(consolidated['season'].unique())}")
print(f"Leagues found: {consolidated['league'].unique()}")
print("\nSample rows:")
print(consolidated.sample(5, random_state=1).to_string())

consolidated.to_csv(OUT, index=False)
print(f"\nSaved to {OUT}")