# ============================================================
# CropSense - Smart Crop Recommendation System
# ============================================================

from pathlib import Path
from textwrap import dedent

import joblib
import pandas as pd
import streamlit as st


# ------------------------------------------------------------
# 1. Page configuration
# ------------------------------------------------------------
st.set_page_config(
    page_title="CropSense | Smart Crop Recommendation",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent


# ------------------------------------------------------------
# 2. Load model files
# ------------------------------------------------------------
@st.cache_resource
def load_model_files():
    model_path = BASE_DIR / "best_crop_model.pkl"
    features_path = BASE_DIR / "crop_features.pkl"
    model_name_path = BASE_DIR / "crop_model_name.pkl"
    scaler_path = BASE_DIR / "crop_scaler.pkl"

    required_files = [model_path, features_path, model_name_path]
    missing_files = [path.name for path in required_files if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(
            "Missing required file(s): " + ", ".join(missing_files)
        )

    model = joblib.load(model_path)
    features = joblib.load(features_path)
    model_name = joblib.load(model_name_path)

    scaler = None
    if model_name in ["Logistic Regression", "SVM"]:
        if not scaler_path.is_file():
            raise FileNotFoundError(
                "The selected model requires crop_scaler.pkl, but it was not found."
            )
        scaler = joblib.load(scaler_path)

    return model, list(features), model_name, scaler


try:
    model, feature_names, model_name, scaler = load_model_files()
except Exception as error:
    st.error(f"Unable to load the trained model: {error}")
    st.info(
        "Keep best_crop_model.pkl, crop_features.pkl, and "
        "crop_model_name.pkl in the same folder as app.py."
    )
    st.stop()


# ------------------------------------------------------------
# 3. Custom styling
# Important: dedent() prevents leading indentation from making
# HTML appear as a Markdown code block.
# ------------------------------------------------------------
st.markdown(
    dedent(
        """
        <style>
        .stApp {
            background-color: #F5F8F2;
        }
        #MainMenu, footer {
            visibility: hidden;
        }
        /* Keep the Streamlit header so the sidebar toggle remains available */
        header[data-testid="stHeader"] {
            background: transparent;
        }
        button[data-testid="collapsedControl"] {
            display: flex !important;
            visibility: visible !important;
        }
        .block-container {
            padding-top: 1rem;
            padding-bottom: 3rem;
            max-width: 1250px;
        }
        section[data-testid="stSidebar"] {
            display: block !important;
            visibility: visible !important;
            background-color: #EAF3E9;
        }
        .hero {
            background: linear-gradient(135deg, #123D2A, #26734D);
            padding: 35px;
            border-radius: 22px;
            color: white;
            margin-bottom: 28px;
            box-shadow: 0 8px 25px rgba(20, 70, 45, 0.12);
        }
        .hero h1 {
            color: white;
            font-size: 38px;
            font-weight: 750;
            margin: 12px 0 10px 0;
        }
        .hero p {
            color: #E0F2E5;
            font-size: 16px;
            line-height: 1.7;
            margin-bottom: 0;
        }
        .hero-tag {
            display: inline-block;
            background: rgba(255, 255, 255, 0.13);
            color: #D5F5DD;
            border: 1px solid rgba(255,255,255,0.2);
            padding: 7px 13px;
            border-radius: 20px;
            font-size: 12px;
        }
        .section-heading {
            color: #174B32;
            font-size: 24px;
            font-weight: 700;
            margin-top: 15px;
            margin-bottom: 5px;
        }
        .section-description {
            color: #64756A;
            font-size: 14px;
            margin-bottom: 22px;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #FFFFFF;
            border: 1px solid #E0E9DF;
            border-radius: 17px;
            padding: 18px;
        }
        label {
            color: #294B38 !important;
            font-weight: 600 !important;
        }
        div[data-testid="stNumberInput"] input {
            border-radius: 9px;
        }
        div.stButton > button, div[data-testid="stFormSubmitButton"] > button {
            width: 100%;
            min-height: 48px;
            border-radius: 12px;
            border: none;
            background: linear-gradient(90deg, #1F7048, #2E8B57);
            color: white;
            font-size: 16px;
            font-weight: 700;
            transition: all 0.2s ease;
        }
        div.stButton > button:hover,
        div[data-testid="stFormSubmitButton"] > button:hover {
            background: #155C39;
            color: white;
            border: none;
            transform: translateY(-1px);
        }
        .result-card {
            background: linear-gradient(135deg, #E2F4E5, #F4FAF2);
            border: 1px solid #B9DEC0;
            border-radius: 20px;
            padding: 28px;
            text-align: center;
            margin-top: 20px;
            margin-bottom: 20px;
        }
        .result-label {
            color: #437353;
            font-size: 14px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1.5px;
        }
        .result-crop {
            color: #174B32;
            font-size: 38px;
            font-weight: 800;
            margin: 12px 0 8px 0;
        }
        .result-description {
            color: #57715E;
            font-size: 14px;
        }
        div[data-testid="stMetric"] {
            background: #FFFFFF;
            border: 1px solid #E0E9DF;
            padding: 17px;
            border-radius: 14px;
        }
        div[data-testid="stMetricLabel"] {
            color: #607365;
        }
        div[data-testid="stMetricValue"] {
            color: #174B32;
        }
        section[data-testid="stSidebar"] {
            background-color: #EAF3E9;
        }
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: #174B32;
        }
        .app-footer {
            text-align: center;
            color: #718176;
            font-size: 13px;
            padding: 25px 0 10px 0;
            border-top: 1px solid #DCE6DC;
            margin-top: 35px;
        }
        </style>
        """
    ),
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# 4. Sidebar
# ------------------------------------------------------------
with st.sidebar:
    st.markdown("# 🌱 CropSense")
    st.caption("SMART AGRICULTURE • MACHINE LEARNING")
    st.divider()

    st.markdown("### About this application")
    st.write(
        "CropSense uses a trained machine learning model to recommend "
        "a crop based on seven soil and environmental measurements."
    )

    st.markdown("### Input parameters")
    st.markdown(
        "- 🧪 Nitrogen (N)\n"
        "- 🧪 Phosphorus (P)\n"
        "- 🧪 Potassium (K)\n"
        "- 🌡️ Temperature\n"
        "- 💧 Humidity\n"
        "- ⚗️ Soil pH\n"
        "- 🌧️ Rainfall"
    )

    st.divider()
    st.markdown("### Model information")
    st.write(f"**Selected model:** {model_name}")
    st.write("**Prediction type:** Multiclass classification")
    st.write(f"**Input features:** {len(feature_names)}")
    st.caption("Predictions depend on the dataset and trained model.")


# ------------------------------------------------------------
# 5. Hero section
# ------------------------------------------------------------
st.markdown(
    dedent(
        """
        <div class="hero">
            <div class="hero-tag">🌿 MACHINE LEARNING FOR AGRICULTURE</div>
            <h1>Find the Right Crop for Your Soil</h1>
            <p>
                Enter your soil nutrients and environmental conditions.
                Our trained machine learning model will analyze your inputs
                and recommend a crop category.
            </p>
        </div>
        """
    ),
    unsafe_allow_html=True,
)

st.info(f"🤖 **Machine Learning Model Used:** {model_name}")

st.markdown(
    '<div class="section-heading">🌾 Crop Recommendation</div>',
    unsafe_allow_html=True,
)
st.markdown(
    dedent(
        """
        <div class="section-description">
            Enter the available soil test and weather measurements to generate a prediction.
        </div>
        """
    ),
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# 6. Input form
# ------------------------------------------------------------
with st.form("crop_prediction_form"):
    st.markdown("### 🧪 Soil Nutrients")
    nutrient_col1, nutrient_col2, nutrient_col3 = st.columns(3)

    with nutrient_col1:
        nitrogen = st.number_input(
            "Nitrogen (N)", min_value=0.0, max_value=300.0,
            value=90.0, step=1.0,
            help="Use the same units as the training dataset.",
        )
    with nutrient_col2:
        phosphorus = st.number_input(
            "Phosphorus (P)", min_value=0.0, max_value=300.0,
            value=42.0, step=1.0,
            help="Use the same units as the training dataset.",
        )
    with nutrient_col3:
        potassium = st.number_input(
            "Potassium (K)", min_value=0.0, max_value=400.0,
            value=43.0, step=1.0,
            help="Use the same units as the training dataset.",
        )

    st.divider()
    st.markdown("### 🌦️ Environmental Conditions")
    env_col1, env_col2 = st.columns(2)

    with env_col1:
        temperature = st.number_input(
            "Temperature (°C)", min_value=-10.0, max_value=60.0,
            value=25.5, step=0.5,
        )
        humidity = st.number_input(
            "Humidity (%)", min_value=0.0, max_value=100.0,
            value=80.0, step=1.0,
        )

    with env_col2:
        ph = st.number_input(
            "Soil pH", min_value=0.0, max_value=14.0,
            value=6.5, step=0.1,
            help="Below 7 is acidic, 7 is neutral, above 7 is alkaline.",
        )
        rainfall = st.number_input(
            "Rainfall (mm)", min_value=0.0, max_value=1000.0,
            value=200.0, step=5.0,
        )

    submitted = st.form_submit_button("🌱 Recommend Suitable Crop")


# ------------------------------------------------------------
# 7. Prediction
# ------------------------------------------------------------
if submitted:
    input_values = {
        "N": nitrogen,
        "P": phosphorus,
        "K": potassium,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall,
    }
    input_df = pd.DataFrame([input_values])

    missing_features = [feature for feature in feature_names if feature not in input_df.columns]

    if missing_features:
        st.error(f"Missing model features: {missing_features}")
    else:
        input_df = input_df[feature_names]

        try:
            processed_input = scaler.transform(input_df) if scaler is not None else input_df
            prediction = model.predict(processed_input)[0]
            display_crop = str(prediction).replace("_", " ").title()

            st.markdown(
                dedent(
                    f"""
                    <div class="result-card">
                        <div class="result-label">Your Model's Prediction</div>
                        <div class="result-crop">🌿 {display_crop}</div>
                        <div class="result-description">
                            Predicted from your supplied soil and environmental measurements.
                        </div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )

            st.markdown("### 📋 Input Summary")
            summary_cols = st.columns(4)
            summary_items = [
                ("Nitrogen", f"{nitrogen:g}"),
                ("Phosphorus", f"{phosphorus:g}"),
                ("Potassium", f"{potassium:g}"),
                ("Temperature", f"{temperature:g} °C"),
                ("Humidity", f"{humidity:g}%"),
                ("Soil pH", f"{ph:g}"),
                ("Rainfall", f"{rainfall:g} mm"),
            ]
            for index, (label, value) in enumerate(summary_items):
                with summary_cols[index % 4]:
                    st.metric(label, value)

            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(processed_input)[0]
                probability_df = pd.DataFrame(
                    {
                        "Crop": [str(c).replace("_", " ").title() for c in model.classes_],
                        "Model Score (%)": probabilities * 100,
                    }
                ).sort_values("Model Score (%)", ascending=False).head(5)

                st.markdown("### 📊 Top 5 Model Predictions")
                st.caption(
                    "Scores are model-estimated class probabilities, not guarantees "
                    "of real-world crop success."
                )
                chart_df = probability_df.set_index("Crop")
                st.bar_chart(chart_df["Model Score (%)"])

                with st.expander("View prediction scores"):
                    st.dataframe(
                        chart_df.style.format({"Model Score (%)": "{:.2f}%"}),
                        use_container_width=True,
                    )

            st.info(
                "🌾 Agricultural note: This is a data-driven prediction, not a guarantee "
                "of crop suitability. Consider local climate, season, soil conditions, "
                "water availability, and advice from an agricultural expert before planting."
            )

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# ------------------------------------------------------------
# 8. Footer
# ------------------------------------------------------------
st.markdown(
    dedent(
        """
        <div class="app-footer">
            <strong>CropSense</strong> — Crop Recommendation System<br><br>
            Powered by Python, Streamlit and Machine Learning.<br><br>
            Built for educational and agricultural decision-support purposes.
        </div>
        """
    ),
    unsafe_allow_html=True,
)
