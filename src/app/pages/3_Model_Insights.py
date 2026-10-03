"""
Model Insights page - real version: live global feature importance,
plus per-player SHAP explanations computed on demand (reusing the
Day 12 approach, but inside the running app now).
"""
import streamlit as st
import sys
import os
import numpy as np
import pandas as pd
import shap

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import load_model, load_player_data, get_feature_columns

st.set_page_config(page_title="Model Insights", page_icon="📊", layout="wide")
st.title("📊 Model Insights")

model, info = load_model()
df = load_player_data()
feature_cols = get_feature_columns(df)

st.write(f"Current model: **{info['name']}**")

POSITION_COLS = {
    "Attack": "position_Attack",
    "Midfield": "position_Midfield",
    "Defender": "position_Defender",
    "Goalkeeper": "position_Goalkeeper",
}
def get_position_name(row):
    return next((pos for pos, col in POSITION_COLS.items() if row[col] == True), "Unknown")


@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

explainer = get_explainer(model)

st.subheader("What drives predictions overall")
importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
st.bar_chart(importances.head(15))
st.caption(
    "Top 15 features by importance. Note: age, experience, and minutes dominate "
    "overall -- this is a known, documented limitation (see README) where defensive "
    "stats help specific cases (like goalkeepers) more than they shift the global ranking."
)

st.divider()

st.subheader("Explain a specific player's prediction")

search_query = st.text_input("Search for a player to explain", placeholder="e.g. Florian Wirtz")

if search_query:
    matches = df[df["name"].str.contains(search_query, case=False, na=False)].sort_values(
        ["name", "season"], ascending=[True, False]
    )
    if matches.empty:
        st.warning("No players found.")
    else:
        matches = matches.copy()
        matches["_label"] = matches.apply(
            lambda r: f"{r['name']} — season {int(r['season'])}, {get_position_name(r)}", axis=1
        )
        choice = st.selectbox("Select player + season", matches["_label"].tolist())
        selected_row = matches[matches["_label"] == choice].iloc[[0]]
        row = selected_row.iloc[0]

        model_input = selected_row[feature_cols]
        pred_log = model.predict(model_input)[0]
        predicted_eur = np.expm1(pred_log)
        actual_eur = row["market_value_eur"]

        col1, col2 = st.columns(2)
        col1.metric("Predicted", f"€{predicted_eur:,.0f}")
        col2.metric("Actual", f"€{actual_eur:,.0f}")

        with st.spinner("Computing explanation..."):
            shap_values = explainer.shap_values(model_input)[0]
            base_log = explainer.expected_value
            base_eur = np.expm1(base_log)

            contributions = pd.Series(shap_values, index=feature_cols)
            contributions_eur = contributions.apply(
                lambda s: np.expm1(base_log + s) - base_eur
            )
            top_contributions = contributions_eur.reindex(
                contributions.abs().sort_values(ascending=False).index
            ).head(10)

        st.caption(f"Base prediction (average player): €{base_eur:,.0f}. "
                    f"Top factors that moved the prediction from there:")
        chart_df = top_contributions.sort_values()
        st.bar_chart(chart_df)
else:
    st.caption("Search for a player above to see what drove their specific prediction.")