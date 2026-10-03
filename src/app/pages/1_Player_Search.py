"""
Player Search page - Day 14: real search, stats display, and live
market value prediction.
"""
import streamlit as st
import sys
import os
import numpy as np

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

search_query = st.text_input("Search for a player", placeholder="e.g. Lamine Yamal")

if not search_query:
    st.caption("Start typing a player's name above.")
    st.stop()

matches = filtered[filtered["name"].str.contains(search_query, case=False, na=False)]

if matches.empty:
    st.warning("No players found matching that name (and league filter, if set).")
    st.stop()

matches = matches.sort_values(["name", "season"], ascending=[True, False])
matches["_position_name"] = matches.apply(get_position_name, axis=1)
matches["_label"] = matches.apply(
    lambda r: f"{r['name']} — season {int(r['season'])}, age {r['age']:.0f}, {r['_position_name']}",
    axis=1
)
choice_label = st.selectbox("Select player + season", matches["_label"].tolist())
selected_row = matches[matches["_label"] == choice_label].iloc[[0]]

row = selected_row.iloc[0]
position_name = get_position_name(row)
league_name = get_league_name(row)
club_match = clubs[clubs["player_id"] == row["player_id"]]
club_name = club_match["current_club_name"].values[0] if not club_match.empty else "Unknown"

st.divider()
st.subheader(row["name"])

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