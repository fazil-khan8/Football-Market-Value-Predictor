"""
Player Search page - adds live search suggestions (via a searchable
dropdown) and player photos, before moving on to Day 15.
"""
import streamlit as st
import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_model, load_player_data, load_club_lookup, get_feature_columns, get_model_input_row

st.set_page_config(page_title="Player Search", page_icon="🔎", layout="wide")
st.title("🔎 Player Search")

model, info = load_model()
df = load_player_data()
clubs = load_club_lookup()
feature_cols = get_feature_columns(df)

LEAGUE_COLS = {
    "Premier League": "league_Premier League",
    "La Liga": "league_La Liga",
    "Bundesliga": "league_Bundesliga",
    "Serie A": "league_Serie A",
    "Ligue 1": "league_Ligue 1",
}
POSITION_COLS = {
    "Attack": "position_Attack",
    "Midfield": "position_Midfield",
    "Defender": "position_Defender",
    "Goalkeeper": "position_Goalkeeper",
}

def get_league_name(row):
    return next((lg for lg, col in LEAGUE_COLS.items() if row[col] == True), "Unknown")

def get_position_name(row):
    return next((pos for pos, col in POSITION_COLS.items() if row[col] == True), "Unknown")

league_choice = st.selectbox("League", ["All"] + list(LEAGUE_COLS.keys()))

filtered = df
if league_choice != "All":
    filtered = filtered[filtered[LEAGUE_COLS[league_choice]] == True]

player_ids_in_scope = filtered["player_id"].unique()
player_display = (
    df[df["player_id"].isin(player_ids_in_scope)][["player_id", "name"]]
    .drop_duplicates("player_id")
    .merge(clubs[["player_id", "current_club_name"]], on="player_id", how="left")
)
player_display["label"] = player_display.apply(
    lambda r: f"{r['name']} ({r['current_club_name'] if pd.notna(r['current_club_name']) else 'Unknown club'})",
    axis=1
)
player_display = player_display.sort_values("name")

selected_label = st.selectbox(
    "Search for a player",
    options=player_display["label"].tolist(),
    index=None,
    placeholder="Start typing a player's name...",
)

if not selected_label:
    st.caption("Start typing a player's name above to see suggestions.")
    st.stop()

selected_player_id = player_display.loc[player_display["label"] == selected_label, "player_id"].values[0]
player_seasons = filtered[filtered["player_id"] == selected_player_id].sort_values("season", ascending=False)

season_labels = player_seasons.apply(
    lambda r: f"Season {int(r['season'])}, age {r['age']:.0f}, {get_position_name(r)}", axis=1
)
season_choice = st.selectbox("Select season", season_labels.tolist())
selected_row = player_seasons.iloc[[season_labels.tolist().index(season_choice)]]

row = selected_row.iloc[0]
position_name = get_position_name(row)
league_name = get_league_name(row)
club_match = clubs[clubs["player_id"] == row["player_id"]]
club_name = club_match["current_club_name"].values[0] if not club_match.empty else "Unknown"
photo_url = club_match["image_url"].values[0] if not club_match.empty else None

st.divider()

photo_col, name_col = st.columns([1, 4])
with photo_col:
    if isinstance(photo_url, str) and photo_url.startswith("http"):
        st.image(photo_url, width=120)
    else:
        st.caption("No photo available")
with name_col:
    st.subheader(row["name"])
    st.caption(f"{position_name} · {club_name} · {league_name}")

info_col1, info_col2, info_col3, info_col4 = st.columns(4)
info_col1.metric("Age (that season)", f"{row['age']:.0f}")
info_col2.metric("Position", position_name)
info_col3.metric("Club (current)", club_name)
info_col4.metric("League", league_name)

stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
stat_col1.metric("Goals", int(row["goals"]))
stat_col2.metric("Assists", int(row["assists"]))
stat_col3.metric("Minutes", f"{int(row['minutes_played']):,}")
stat_col4.metric("Appearances", int(row["appearances_count"]))

if row.get("has_advanced_stats", 0) == 1:
    st.caption("Advanced stats available for this season:")
    adv_col1, adv_col2, adv_col3, adv_col4 = st.columns(4)
    adv_col1.metric("Tackles won", int(row["tackles_won"]))
    adv_col2.metric("Interceptions", int(row["interceptions"]))
    adv_col3.metric("Clearances", int(row["clearances"]))
    if position_name == "Goalkeeper":
        adv_col4.metric("Clean sheets", int(row["clean_sheets"]))

st.divider()
model_input = get_model_input_row(selected_row, feature_cols)
pred_log = model.predict(model_input)[0]
predicted_eur = np.expm1(pred_log)
actual_eur = row["market_value_eur"]

pred_col1, pred_col2 = st.columns(2)
pred_col1.metric("💰 Predicted Market Value", f"€{predicted_eur:,.0f}")
pred_col2.metric("Actual Market Value", f"€{actual_eur:,.0f}",
                  delta=f"{(predicted_eur - actual_eur) / actual_eur * 100:+.0f}% vs predicted")