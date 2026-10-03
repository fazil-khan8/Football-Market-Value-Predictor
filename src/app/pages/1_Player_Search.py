"""
Player Search page (structure only for Day 13 -- real search and
prediction logic gets built on Day 14).
"""
import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_model, load_player_data

st.set_page_config(page_title="Player Search", page_icon="🔎", layout="wide")
st.title("🔎 Player Search")

model, info = load_model()
df = load_player_data()

search_query = st.text_input("Search for a player", placeholder="e.g. Lamine Yamal")

if search_query:
    st.info("Search and prediction logic will be built out on Day 14.")
    matches = df[df["name"].str.contains(search_query, case=False, na=False)]
    st.write(f"Found {matches['name'].nunique()} matching player(s) in the database:")
    st.dataframe(matches[["name", "season"]].drop_duplicates().head(20))
else:
    st.caption("Start typing a player's name above.")