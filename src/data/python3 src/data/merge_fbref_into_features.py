"""
Extension, step 2 - Fuzzy-match the consolidated FBref defensive/
goalkeeper stats onto our existing feature table by player name,
within the same league and season (keeps matching fast and accurate
by shrinking the candidate pool).

Rows that don't get a confident match (different seasons than FBref
covers, or a name that didn't match closely enough) get 0 for the new
stats and has_advanced_stats=0, so the model can still tell "genuinely
zero tackles" apart from "no data available" if it's useful.

Run from the project root:
    python3 src/features/merge_fbref_into_features.py
"""
import pandas as pd
import numpy as np
from rapidfuzz import process, fuzz
from unidecode import unidecode

PROCESSED = "data/processed"
RAW = "data/raw/fbref"

features = pd.read_csv(f"{PROCESSED}/player_seasons_features.csv")
fbref = pd.read_csv(f"{RAW}/fbref_consolidated.csv")

def normalize(name):
    return unidecode(str(name)).lower().strip()

features["name_norm"] = features["name"].apply(normalize)
fbref["player_norm"] = fbref["player"].apply(normalize)

league_cols = [c for c in features.columns if c.startswith("league_")]
features["league"] = features[league_cols].idxmax(axis=1).str.replace("league_", "")

stat_cols = ["tackles", "tackles_won", "interceptions", "clearances", "blocks",
             "errors", "goals_against", "saves", "save_pct", "clean_sheets"]

for col in stat_cols:
    features[col] = np.nan
features["has_advanced_stats"] = 0

MATCH_THRESHOLD = 87

matched_count = 0
for (league, season), group in features.groupby(["league", "season"]):
    candidates = fbref[(fbref["league"] == league) & (fbref["season"] == season)]
    if candidates.empty:
        continue
    choices = candidates["player_norm"].tolist()
    for idx, row in group.iterrows():
        result = process.extractOne(row["name_norm"], choices, scorer=fuzz.WRatio, score_cutoff=MATCH_THRESHOLD)
        if result:
            _, score, pos_in_list = result
            matched_row = candidates.iloc[pos_in_list]
            for col in stat_cols:
                features.at[idx, col] = matched_row[col]
            features.at[idx, "has_advanced_stats"] = 1
            matched_count += 1

print(f"Matched {matched_count:,} / {len(features):,} rows ({matched_count/len(features)*100:.1f}%)")
print("\nMatch rate by season (only seasons FBref covers will be non-zero):")
print(features.groupby("season")["has_advanced_stats"].mean().round(2))

for col in stat_cols:
    features[col] = features[col].fillna(0)

features["tackles_per_90"] = features["tackles"] / features["minutes_played"] * 90
features["interceptions_per_90"] = features["interceptions"] / features["minutes_played"] * 90
features["clearances_per_90"] = features["clearances"] / features["minutes_played"] * 90
features["blocks_per_90"] = features["blocks"] / features["minutes_played"] * 90
features["defensive_actions_per_90"] = (
    features["tackles_per_90"] + features["interceptions_per_90"] + features["clearances_per_90"]
)

features = features.drop(columns=["name_norm", "league"])
features.to_csv(f"{PROCESSED}/player_seasons_features_v2.csv", index=False)

print(f"\nSaved to {PROCESSED}/player_seasons_features_v2.csv")
print(f"New shape: {features.shape}")
print("\nStats summary (matched rows only, has_advanced_stats=1):")
print(features[features["has_advanced_stats"] == 1][stat_cols].describe().round(2))