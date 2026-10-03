"""
Custom Prediction page - Day 15: real form, builds the full feature
row the model expects, and returns a prediction.
"""
import streamlit as st
import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_model, load_player_data, get_feature_columns

st.set_page_config(page_title="Custom Prediction", page_icon="✍️", layout="wide")
st.title("✍️ Custom Player Prediction")
st.caption("Enter a hypothetical player's stats to get a predicted market value.")

model, info = load_model()
df = load_player_data()
feature_cols = get_feature_columns(df)

LEAGUES = ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"]
POSITIONS = ["Attack", "Midfield", "Defender", "Goalkeeper"]
FEET = ["right", "left", "both", "unknown"]

with st.form("custom_prediction_form"):
    st.subheader("Player profile")
    col1, col2, col3 = st.columns(3)
    with col1:
        age = col1.number_input("Age", min_value=15, max_value=45, value=22)
        league = col1.selectbox("League", LEAGUES)
    with col2:
        position = col2.selectbox("Position", POSITIONS)
        foot = col2.selectbox("Preferred foot", FEET)
    with col3:
        height_cm = col3.number_input("Height (cm)", min_value=150, max_value=210, value=180)
        seasons_exp = col3.number_input("Seasons of experience", min_value=0, max_value=25, value=3)

    st.subheader("Playing time & output")
    col4, col5, col6 = st.columns(3)
    with col4:
        appearances = col4.number_input("Appearances", min_value=0, value=30)
        minutes = col4.number_input("Minutes played", min_value=0, value=2500)
    with col5:
        goals = col5.number_input("Goals", min_value=0, value=10)
        assists = col5.number_input("Assists", min_value=0, value=5)
    with col6:
        yellow_cards = col6.number_input("Yellow cards", min_value=0, value=3)
        red_cards = col6.number_input("Red cards", min_value=0, value=0)

    tackles_won = interceptions = clearances = blocks = errors = 0
    goals_against = saves = save_pct = clean_sheets = 0

    if position in ("Defender", "Midfield"):
        st.subheader("Defensive stats")
        d1, d2, d3, d4 = st.columns(4)
        tackles_won = d1.number_input("Tackles won", min_value=0, value=20)
        interceptions = d2.number_input("Interceptions", min_value=0, value=15)
        clearances = d3.number_input("Clearances", min_value=0, value=10)
        blocks = d4.number_input("Blocks", min_value=0, value=5)

    if position == "Goalkeeper":
        st.subheader("Goalkeeper stats")
        g1, g2, g3, g4 = st.columns(4)
        goals_against = g1.number_input("Goals against", min_value=0, value=30)
        saves = g2.number_input("Saves", min_value=0, value=80)
        save_pct = g3.number_input("Save %", min_value=0.0, max_value=100.0, value=70.0)
        clean_sheets = g4.number_input("Clean sheets", min_value=0, value=10)

    submitted = st.form_submit_button("Predict Market Value", use_container_width=True)

if submitted:
    row = {col: 0 for col in feature_cols}

    row["age"] = age
    row["age_squared"] = age ** 2
    row["height_in_cm"] = height_cm
    row["seasons_of_experience"] = seasons_exp
    row["appearances_count"] = appearances
    row["minutes_played"] = minutes
    row["minutes_per_appearance"] = minutes / appearances if appearances > 0 else 0
    row["goals"] = goals
    row["assists"] = assists
    row["goals_per_90"] = goals / minutes * 90 if minutes > 0 else 0
    row["assists_per_90"] = assists / minutes * 90 if minutes > 0 else 0
    row["goal_contributions_per_90"] = row["goals_per_90"] + row["assists_per_90"]
    row["yellow_cards"] = yellow_cards
    row["red_cards"] = red_cards

    row[f"league_{league}"] = 1
    row[f"position_{position}"] = 1
    row[f"foot_{foot}"] = 1

    row["tackles_won"] = tackles_won
    row["interceptions"] = interceptions
    row["clearances"] = clearances
    row["blocks"] = blocks
    row["errors"] = errors
    row["goals_against"] = goals_against
    row["saves"] = saves
    row["save_pct"] = save_pct
    row["clean_sheets"] = clean_sheets
    row["has_advanced_stats"] = 1

    row["tackles_per_90"] = tackles_won / minutes * 90 if minutes > 0 else 0
    row["interceptions_per_90"] = interceptions / minutes * 90 if minutes > 0 else 0
    row["clearances_per_90"] = clearances / minutes * 90 if minutes > 0 else 0
    row["blocks_per_90"] = blocks / minutes * 90 if minutes > 0 else 0
    row["defensive_actions_per_90"] = (
        row["tackles_per_90"] + row["interceptions_per_90"] + row["clearances_per_90"]
    )

    input_df = pd.DataFrame([row]).reindex(columns=feature_cols, fill_value=0)

    pred_log = model.predict(input_df)[0]
    predicted_eur = np.expm1(pred_log)

    st.divider()
    st.metric("💰 Estimated Market Value", f"€{predicted_eur:,.0f}")

    with st.expander("See the exact feature values sent to the model"):
        st.dataframe(input_df.T.rename(columns={0: "value"}))