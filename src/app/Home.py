"""
Home page - Day 16: card-style layout + a "most valuable players"
chart instead of a plain bullet list.
"""
import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import load_model, load_player_data

st.set_page_config(
    page_title="Football Market Value Predictor",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ AI Football Player Market Value Predictor")
st.markdown(
    "Predicts a football player's market value from playing statistics, "
    "covering Europe's top 5 leagues: Premier League, La Liga, Bundesliga, "
    "Serie A, and Ligue 1."
)

with st.spinner("Loading model and player data..."):
    model, info = load_model()
    df = load_player_data()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Players in database", f"{df['name'].nunique():,}")
col2.metric("Player-seasons", f"{len(df):,}")
col3.metric("Seasons covered", f"{int(df['season'].min())}–{int(df['season'].max())}")
col4.metric("Model", info["name"])

st.divider()

card1, card2, card3 = st.columns(3)
with card1:
    with st.container(border=True):
        st.markdown("### 🔎 Player Search")
        st.write("Look up a real player and see their predicted market value, "
                 "alongside their real stats and value history.")
with card2:
    with st.container(border=True):
        st.markdown("### ✍️ Custom Prediction")
        st.write("Enter a hypothetical player's stats -- age, league, position, "
                 "goals, assists -- and get an estimated market value.")
with card3:
    with st.container(border=True):
        st.markdown("### 📊 Model Insights")
        st.write("See what drives the model's predictions overall, and explain "
                 "any individual player's predicted value.")

st.divider()

st.subheader("💰 Most valuable players in the database")
top_players = (
    df.sort_values("market_value_eur", ascending=False)
    .drop_duplicates("player_id")
    .head(10)
    .set_index("name")["market_value_eur"]
    / 1_000_000
)
st.bar_chart(top_players)
st.caption("Highest single-season market value on record for each player, in €M.")