"""
Shared utilities for the Streamlit app -- loading the trained model
and player data once, cached, so every page can reuse them without
re-reading files on every interaction.
"""
import streamlit as st
import pandas as pd
import joblib
import json

MODELS_DIR = "models"
PROCESSED_DIR = "data/processed"

DROP_COLS = ["player_id", "season", "name", "market_value_eur", "log_market_value_eur"]


@st.cache_resource
def load_model():
    """Load the trained model + its metadata (name, whether it needs scaling)."""
    model = joblib.load(f"{MODELS_DIR}/best_model.pkl")
    with open(f"{MODELS_DIR}/best_model_info.json") as f:
        info = json.load(f)
    return model, info


@st.cache_data
def load_player_data():
    """Load the full player-season feature table (v2, with defensive/
    keeper stats) used for both search and training."""
    df = pd.read_csv(f"{PROCESSED_DIR}/player_seasons_features_v2.csv")
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """The feature columns to feed the model, in the same shape used
    during training (everything except id/name/target columns)."""
    return [c for c in df.columns if c not in DROP_COLS]


def predict_value(model, info, feature_row: pd.DataFrame):
    """Run one row of features through the model, return predicted
    market value in euros (converting back from the log scale)."""
    import numpy as np
    if info.get("needs_scaling"):
        raise NotImplementedError("Scaling path not needed for the current model.")
    pred_log = model.predict(feature_row)[0]
    return float(np.expm1(pred_log))