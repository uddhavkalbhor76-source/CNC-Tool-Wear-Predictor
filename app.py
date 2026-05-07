import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
st.set_page_config(
    page_title="Tool Wear Predictor",
    page_icon="🔧",
    layout="centered"
)


@st.cache_data   
def generate_dataset(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a synthetic CNC tool wear dataset using NumPy distributions.
    Healthy tools show lower vibration, temperature, and surface roughness.
    Worn/defective tools exhibit higher and noisier sensor readings.
    """
    rng = np.random.default_rng(random_state)

    n_healthy = n_samples // 2
    healthy = pd.DataFrame({
        "Vibration_mm_s":       rng.normal(loc=2.5,  scale=0.5,  size=n_healthy),
        "Temperature_C":        rng.normal(loc=55,   scale=5,    size=n_healthy),
        "Cutting_Load_N":       rng.normal(loc=200,  scale=20,   size=n_healthy),
        "Spindle_Speed_RPM":    rng.normal(loc=3000, scale=150,  size=n_healthy),
        "Surface_Roughness_um": rng.normal(loc=1.2,  scale=0.2,  size=n_healthy),
        "Tool_Status": 0   
    })

    n_worn = n_samples - n_healthy
    worn = pd.DataFrame({
        "Vibration_mm_s":       rng.normal(loc=6.5,  scale=1.2,  size=n_worn),
        "Temperature_C":        rng.normal(loc=90,   scale=10,   size=n_worn),
        "Cutting_Load_N":       rng.normal(loc=350,  scale=40,   size=n_worn),
        "Spindle_Speed_RPM":    rng.normal(loc=2600, scale=200,  size=n_worn),
        "Surface_Roughness_um": rng.normal(loc=3.8,  scale=0.6,  size=n_worn),
        "Tool_Status": 1  
    })

    df = pd.concat([healthy, worn], ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True) 
    df = df.clip(lower=0)  
    return df



@st.cache_resource  
def train_model(df: pd.DataFrame):
    """
    Splits data, scales features, and trains a Random Forest Classifier.
    Random Forest is an ensemble of Decision Trees — robust and interpretable.
    Returns: model, scaler, accuracy, feature names, classification report
    """
    feature_cols = [
        "Vibration_mm_s", "Temperature_C", "Cutting_Load_N",
        "Spindle_Speed_RPM", "Surface_Roughness_um"
    ]
    X = df[feature_cols]
    y = df["Tool_Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_sc, y_train)

    y_pred   = model.predict(X_test_sc)
    accuracy = accuracy_score(y_test, y_pred)
    report   = classification_report(y_test, y_pred, target_names=["Healthy", "Defective"])

    return model, scaler, accuracy, feature_cols, report




def main():
    st.title("CNC Tool Wear Predictor")
    st.markdown(
        """
        **Course:** MDM-271-MEC — AI & ML in Mechanical Engineering  
        **College:** NMIET | SPPU 2024 Pattern  
        This app uses a **Random Forest Classifier** trained on synthetic CNC sensor
        data to predict whether a cutting tool is **Healthy ** or **Defective **.
        """
    )
    st.divider()

    df = generate_dataset()
    model, scaler, accuracy, feature_cols, report = train_model(df)

    with st.sidebar:
        st.header("📊 Model Info")
        st.metric("Algorithm", "Random Forest")
        st.metric("Training Samples", "800")
        st.metric("Test Samples", "200")
        st.metric("Test Accuracy", f"{accuracy * 100:.2f}%")
        st.divider()
        st.subheader("Classification Report")
        st.code(report)
        st.divider()
        st.caption("MDM-271-MEC | NMIET | SPPU 2024")

    with st.expander("View Sample Training Data (first 10 rows)"):
        st.dataframe(
            df.head(10).rename(columns={"Tool_Status": "Status (0=OK, 1=Worn)"}),
            use_container_width=True
        )
        st.caption(f"Full dataset: {len(df)} rows × {df.shape[1]} columns")

    st.subheader("Enter Live Sensor Readings")
    st.markdown("Adjust the sliders below to simulate real-time CNC sensor data:")

    col1, col2 = st.columns(2)

    with col1:
        vibration = st.slider(
            "Vibration (mm/s)",
            min_value=0.5, max_value=10.0, value=3.0, step=0.1,
            help="Higher vibration → tool may be worn"
        )
        temperature = st.slider(
            "Temperature (°C)",
            min_value=30.0, max_value=130.0, value=60.0, step=1.0,
            help="Worn tools generate more heat due to friction"
        )
        cutting_load = st.slider(
            "Cutting Load (N)",
            min_value=100.0, max_value=500.0, value=220.0, step=5.0,
            help="Worn tools require higher cutting force"
        )

    with col2:
        spindle_speed = st.slider(
            "Spindle Speed (RPM)",
            min_value=1500.0, max_value=4000.0, value=3000.0, step=50.0,
            help="Effective cutting speed"
        )
        surface_roughness = st.slider(
            "Surface Roughness (µm)",
            min_value=0.5, max_value=6.0, value=1.5, step=0.1,
            help="Higher Ra value → poor finish → possible tool wear"
        )

    st.divider()
    st.subheader("Prediction")

    if st.button("Predict Tool Health", type="primary", use_container_width=True):

        input_data = np.array([[vibration, temperature, cutting_load,
                                 spindle_speed, surface_roughness]])
        input_scaled = scaler.transform(input_data)

        prediction   = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]

        healthy_prob   = probabilities[0] * 100
        defective_prob = probabilities[1] * 100

        if prediction == 0:
            st.success("### Tool Status: HEALTHY")
            st.markdown(
                f"""
                The Random Forest model classified this tool as **Healthy** with
                **{healthy_prob:.1f}% confidence**.

                **What this means:**  
                The sensor readings (Vibration: {vibration} mm/s, Temperature: {temperature}°C,
                Load: {cutting_load} N, Speed: {spindle_speed} RPM, Roughness: {surface_roughness} µm)
                fall within the **normal operating range** for a fresh or slightly used tool.  
                No immediate maintenance action is required.
                """
            )
        else:
            st.error("### Tool Status: DEFECTIVE (Tool Wear Detected)")
            st.markdown(
                f"""
                The Random Forest model classified this tool as **Defective/Worn** with
                **{defective_prob:.1f}% confidence**.

                **What this means:**  
                One or more sensor values — likely Vibration ({vibration} mm/s),
                Temperature ({temperature}°C), or Surface Roughness ({surface_roughness} µm) —
                are **outside healthy operating bounds**.  
                **Recommended Action:** Inspect and replace the cutting tool immediately
                to avoid scrap parts or machine damage.
                """
            )

        st.markdown("**Confidence Breakdown:**")
        conf_df = pd.DataFrame({
            "Status":      ["Healthy", "Defective"],
            "Confidence %": [healthy_prob, defective_prob]
        })
        st.bar_chart(conf_df.set_index("Status"), color=["#2ecc71"])

    st.divider()
    st.subheader(" Feature Importance (What influences the model most?)")

    importances = model.feature_importances_
    feat_labels = ["Vibration", "Temperature", "Cutting Load", "Spindle Speed", "Surface Roughness"]

    fig, ax = plt.subplots(figsize=(7, 3.5))
    colors = ["#e74c3c" if i == np.argmax(importances) else "#3498db"
              for i in range(len(importances))]
    bars = ax.barh(feat_labels, importances * 100, color=colors)
    ax.set_xlabel("Importance (%)")
    ax.set_title("Random Forest — Feature Importance")
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.invert_yaxis()
    plt.tight_layout()
    st.pyplot(fig)
    st.caption(
        "The highlighted bar (red) is the most important feature for prediction. "
        "This is determined by how much each feature reduces impurity across all trees."
    )

    st.divider()
    with st.expander(" How does Random Forest work? (For viva explanation)"):
        st.markdown(
            """
            **Random Forest** is an *ensemble learning* algorithm from **Unit II** of MDM-271-MEC.
"""
        )


if __name__ == "__main__":
    main()
