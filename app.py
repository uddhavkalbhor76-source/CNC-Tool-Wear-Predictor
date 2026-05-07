import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# --- COLOR PALETTE (Industrial "Midnight Forge") ---
PRIMARY_COLOR = "#00FFAA"  # Neon Mint
BG_SECONDARY = "#1E2630"   # Deep Slate
ACCENT_RED = "#FF4B4B"    # Error/Warning
ACCENT_BLUE = "#3498DB"   # Info
TEXT_GRAY = "#A9B1D6"

st.set_page_config(
    page_title="Tool Wear Predictor",
    page_icon="🔧",
    layout="centered"
)

@st.cache_data   
def generate_dataset(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    n_half = n_samples // 2
    
    # Vectorized data generation for better speed
    data = {
        "Vibration_mm_s": np.concatenate([rng.normal(2.5, 0.5, n_half), rng.normal(6.5, 1.2, n_samples - n_half)]),
        "Temperature_C": np.concatenate([rng.normal(55, 5, n_half), rng.normal(90, 10, n_samples - n_half)]),
        "Cutting_Load_N": np.concatenate([rng.normal(200, 20, n_half), rng.normal(350, 40, n_samples - n_half)]),
        "Spindle_Speed_RPM": np.concatenate([rng.normal(3000, 150, n_half), rng.normal(2600, 200, n_samples - n_half)]),
        "Surface_Roughness_um": np.concatenate([rng.normal(1.2, 0.2, n_half), rng.normal(3.8, 0.6, n_samples - n_half)]),
        "Tool_Status": np.concatenate([np.zeros(n_half), np.ones(n_samples - n_half)])
    }
    
    df = pd.DataFrame(data).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df.clip(lower=0)

@st.cache_resource  
def train_model(df: pd.DataFrame):
    feature_cols = ["Vibration_mm_s", "Temperature_C", "Cutting_Load_N", "Spindle_Speed_RPM", "Surface_Roughness_um"]
    X, y = df[feature_cols], df["Tool_Status"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train_sc, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test_sc))
    report = classification_report(y_test, model.predict(X_test_sc), target_names=["Healthy", "Defective"])

    return model, scaler, accuracy, feature_cols, report

def main():
    st.title("🔧 CNC Tool Wear Predictor")
    st.markdown(f"<p style='color:{TEXT_GRAY}'><b>Course:</b> MDM-271-MEC — AI & ML in Mechanical Engineering<br><b>College:</b> NMIET | SPPU 2024 Pattern</p>", unsafe_allow_html=True)
    st.divider()

    df = generate_dataset()
    model, scaler, accuracy, feature_cols, report = train_model(df)

    with st.sidebar:
        st.header("📊 Model Info")
        st.metric("Algorithm", "Random Forest", delta="Ensemble")
        st.metric("Test Accuracy", f"{accuracy * 100:.2f}%")
        with st.expander("Classification Report"):
            st.code(report)
        st.caption("MDM-271-MEC | NMIET | SPPU 2024")

    # Input Section
    st.subheader("Enter Live Sensor Readings")
    col1, col2 = st.columns(2)
    with col1:
        vibration = st.slider("Vibration (mm/s)", 0.5, 10.0, 3.0, 0.1)
        temp = st.slider("Temperature (°C)", 30.0, 130.0, 60.0, 1.0)
        load = st.slider("Cutting Load (N)", 100.0, 500.0, 220.0, 5.0)
    with col2:
        speed = st.slider("Spindle Speed (RPM)", 1500.0, 4000.0, 3000.0, 50.0)
        rough = st.slider("Surface Roughness (µm)", 0.5, 6.0, 1.5, 0.1)

    if st.button("Predict Tool Health", type="primary", use_container_width=True):
        input_sc = scaler.transform([[vibration, temp, load, speed, rough]])
        pred = model.predict(input_sc)[0]
        probs = model.predict_proba(input_sc)[0]

        if pred == 0:
            st.success(f"### Tool Status: HEALTHY ({probs[0]*100:.1f}% Confidence)")
        else:
            st.error(f"### Tool Status: DEFECTIVE ({probs[1]*100:.1f}% Confidence)")
        
        # Confidence Chart
        conf_df = pd.DataFrame({"Confidence %": [probs[0]*100, probs[1]*100]}, index=["Healthy", "Defective"])
        st.bar_chart(conf_df, color=PRIMARY_COLOR)

    # Feature Importance Visualization
    st.divider()
    st.subheader("Feature Importance")
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#0E1117') # Match Streamlit Dark Theme
    ax.set_facecolor('#0E1117')
    
    importances = model.feature_importances_
    labels = ["Vibration", "Temp", "Load", "Speed", "Roughness"]
    colors = [ACCENT_RED if i == max(importances) else ACCENT_BLUE for i in importances]
    
    bars = ax.barh(labels, importances * 100, color=colors)
    ax.bar_label(bars, fmt="%.1f%%", color="white", padding=3)
    ax.tick_params(colors='white')
    st.pyplot(fig)

if __name__ == "__main__":
    main()
