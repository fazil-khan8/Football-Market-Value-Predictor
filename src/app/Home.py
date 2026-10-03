"""
Home page - app entry point.

Run the whole app with:
    streamlit run src/app/Home.py
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

st.success("Model and data loaded.")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Players in database", f"{df['name'].nunique():,}")
col2.metric("Player-seasons", f"{len(df):,}")
col3.metric("Seasons covered", f"{int(df['season'].min())}–{int(df['season'].max())}")
col4.metric("Model", info["name"])

st.divider()

st.markdown(
    """
    ### What you can do here
    - **🔎 Player Search** — look up a real player and see their predicted market value.
    - **✍️ Custom Prediction** — enter a hypothetical player's stats and get an estimate.
    - **📊 Model Insights** — see what drives the model's predictions.

    Use the sidebar to navigate between pages.
    """
)