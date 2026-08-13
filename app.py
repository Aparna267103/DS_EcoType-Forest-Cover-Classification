# ========================= app.py =========================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ================== LOAD ==================
model = joblib.load("forest_model.pkl")
le = joblib.load("label_encoder.pkl")

# Try to load a saved scaler; if missing, fit one from the training CSV as a fallback
scaler = joblib.load("scaler.pkl")
try:
        joblib.dump(scaler, "scaler.pkl")
except Exception:
        pass

st.set_page_config(page_title="Forest Cover Prediction")

st.title("🌲 EcoType: Forest Cover Prediction")

st.write("Enter the forest geographical details")


# ================== USER INPUTS ==================

Elevation = st.number_input("Elevation", value=2500.0)

Aspect = st.number_input("Aspect", value=100.0)

Slope = st.number_input("Slope", value=20.0)

Horizontal_Distance_To_Hydrology = st.number_input(
    "Horizontal Distance To Hydrology",
    value=100.0
)

Vertical_Distance_To_Hydrology = st.number_input(
    "Vertical Distance To Hydrology",
    value=20.0
)

Horizontal_Distance_To_Roadways = st.number_input(
    "Horizontal Distance To Roadways",
    value=200.0
)

Hillshade_9am = st.number_input(
    "Hillshade 9am",
    value=220.0
)

Hillshade_Noon = st.number_input(
    "Hillshade Noon",
    value=230.0
)

Hillshade_3pm = st.number_input(
    "Hillshade 3pm",
    value=210.0
)

Horizontal_Distance_To_Fire_Points = st.number_input(
    "Horizontal Distance To Fire Points",
    value=300.0
)

Wilderness_Area = st.selectbox(
    "Wilderness Area",
    [1, 2, 3, 4]
)

Soil_Type = st.selectbox(
    "Soil Type",
    list(range(1, 41))
)


# ================== PREDICT ==================
if st.button("Predict Cover Type"):

    # ================== SKEWNESS TRANSFORMATION ==================
    Elevation = np.log1p(Elevation)

    Horizontal_Distance_To_Hydrology = np.log1p(
        Horizontal_Distance_To_Hydrology
    )

    Horizontal_Distance_To_Roadways = np.log1p(
        Horizontal_Distance_To_Roadways
    )

    Horizontal_Distance_To_Fire_Points = np.log1p(
        Horizontal_Distance_To_Fire_Points
    )

    # ================== FEATURE ENGINEERING ==================
    Hydrology_Distance = (
        Horizontal_Distance_To_Hydrology +
        Vertical_Distance_To_Hydrology
    )

    Hillshade_mean = (
        Hillshade_9am +
        Hillshade_Noon +
        Hillshade_3pm
    ) / 3

    # ================== CREATE INPUT DATAFRAME ==================
    input_data = pd.DataFrame({

        'Elevation': [Elevation],

        'Aspect': [Aspect],

        'Slope': [Slope],

        'Horizontal_Distance_To_Hydrology': [
            Horizontal_Distance_To_Hydrology
        ],

        'Vertical_Distance_To_Hydrology': [
            Vertical_Distance_To_Hydrology
        ],

        'Horizontal_Distance_To_Roadways': [
            Horizontal_Distance_To_Roadways
        ],

        'Hillshade_9am': [Hillshade_9am],

        'Hillshade_Noon': [Hillshade_Noon],

        'Hillshade_3pm': [Hillshade_3pm],

        'Horizontal_Distance_To_Fire_Points': [
            Horizontal_Distance_To_Fire_Points
        ],

        'Wilderness_Area': [Wilderness_Area],

        'Soil_Type': [Soil_Type],

        'Hydrology_Distance': [Hydrology_Distance],

        'Hillshade_mean': [Hillshade_mean]
    })

    # ================== PREDICTION ==================
    # scale input to match training preprocessing
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    output = le.inverse_transform(prediction)

    st.success(f"🌳 Predicted Forest Cover Type: {output[0]}")
