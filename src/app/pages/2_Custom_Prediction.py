"""
Custom Prediction page (structure only for Day 13 -- real input form
and prediction logic gets built on Day 15).
"""
import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_model

st.set_page_config(page_title="Custom Prediction", page_icon="✍️", layout="wide")
st.title("✍️ Custom Player Prediction")
st.caption("Enter a hypothetical player's stats to get a predicted market value.")

model, info = load_model()

col1, col2 = st.columns(2)
with col1:
    st.number_input("Age", min_value=15, max_value=45, value=22)
    st.selectbox("League", ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1"])
    st.selectbox("Position", ["Attack", "Midfield", "Defender", "Goalkeeper"])
with col2:
    st.number_input("Minutes played", min_value=0, value=2000)
    st.number_input("Goals", min_value=0, value=10)
    st.number_input("Assists", min_value=0, value=5)

st.button("Predict Market Value", disabled=True, help="Full prediction logic coming on Day 15")
st.info("This form is a placeholder -- full input fields and prediction logic will be built on Day 15.")