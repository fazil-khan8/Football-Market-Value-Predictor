"""
Model Insights page (structure only for Day 13 -- real charts and
SHAP explanations get built out later, reusing the work from Day 12).
"""
import streamlit as st
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_model

st.set_page_config(page_title="Model Insights", page_icon="📊", layout="wide")
st.title("📊 Model Insights")

model, info = load_model()
st.write(f"Current model: **{info['name']}**")

st.info(
    "This page will show feature importance and SHAP explanations "
    "(reusing the analysis from Day 12) once the app's visual layer "
    "is built out."
)

shap_path = "reports/figures/shap_summary_v2.png"
if os.path.exists(shap_path):
    st.image(shap_path, caption="Global feature importance (from Day 12 analysis)")