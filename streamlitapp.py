import streamlit as st
import joblib
import pandas as pd

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Sleep Disorder Predictor",
    page_icon="😴",
    layout="wide"
)

# -------------------------------
# CACHE MODEL LOADING
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("xgb_sleep_model.pkl")

model = load_model()

# -------------------------------
# APP TITLE
# -------------------------------
st.title("😴 Sleep Disorder Prediction App")
st.markdown("Get predictions + detailed health insights based on your lifestyle.")

# -------------------------------
# SIDEBAR INPUTS
# -------------------------------
st.sidebar.header("📊 Enter Your Health Details")

age = st.sidebar.slider("🧑 Age", 10, 100, 25)
sleep_duration = st.sidebar.slider("🛌 Sleep Duration (hours)", 0, 12, 7)
quality_of_sleep = st.sidebar.slider("🌙 Quality of Sleep (1=Poor, 10=Excellent)", 1, 10, 5)
physical_activity_level = st.sidebar.slider("🏃 Physical Activity Level (minutes/day)", 0, 300, 30)
stress_level = st.sidebar.slider("😥 Stress Level (1=Low, 10=High)", 1, 10, 5)
heart_rate = st.sidebar.number_input("❤️ Heart Rate (bpm)", min_value=40, max_value=120, value=70)
daily_steps = st.sidebar.number_input("👣 Daily Steps", min_value=0, max_value=30000, value=5000)
bp_systolic = st.sidebar.number_input("💓 Blood Pressure - Systolic", min_value=80, max_value=200, value=120)
bp_diastolic = st.sidebar.number_input("💓 Blood Pressure - Diastolic", min_value=50, max_value=130, value=80)

# -------------------------------
# FEATURE PACKING
# -------------------------------
features = [[
    age,
    sleep_duration,
    quality_of_sleep,
    physical_activity_level,
    stress_level,
    heart_rate,
    daily_steps,
    bp_systolic,
    bp_diastolic
]]

# -------------------------------
# PREDICTION
# -------------------------------
if st.sidebar.button("🔍 Predict Sleep Disorder"):
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0]  # prediction probabilities

    st.subheader("📌 Prediction Result")
    st.success(f"🧾 **{prediction}** (Model Confidence: {max(proba)*100:.2f}%)")

    # -------------------------------
    # HEALTH RISK DASHBOARD
    # -------------------------------
    st.subheader("📊 Health Risk Dashboard")

    col1, col2, col3 = st.columns(3)
    with col1:
        if stress_level > 7:
            st.error("😥 High Stress")
        elif stress_level > 4:
            st.warning("😐 Moderate Stress")
        else:
            st.success("😊 Low Stress")

    with col2:
        if sleep_duration < 5:
            st.error("🛌 Poor Sleep")
        elif sleep_duration < 7:
            st.warning("😴 Borderline Sleep")
        else:
            st.success("🌙 Healthy Sleep")

    with col3:
        if bp_systolic > 140 or bp_diastolic > 90:
            st.error("💓 High Blood Pressure")
        else:
            st.success("✅ Normal Blood Pressure")

    # -------------------------------
    # PERSONALIZED SUGGESTIONS
    # -------------------------------
    st.subheader("💡 Personalized Suggestions")
    suggestions = []
    if stress_level > 7:
        suggestions.append("Try relaxation techniques (meditation, deep breathing).")
    if sleep_duration < 6:
        suggestions.append("Aim for at least 7-8 hours of quality sleep.")
    if physical_activity_level < 30:
        suggestions.append("Increase daily activity — at least 30 mins/day.")
    if heart_rate > 100:
        suggestions.append("Monitor your heart health, consider a check-up.")
    if bp_systolic > 140:
        suggestions.append("Consult doctor for high blood pressure management.")

    if suggestions:
        for s in suggestions:
            st.markdown(f"- {s}")
    else:
        st.success("✅ Your lifestyle looks balanced! Keep it up.")

    # -------------------------------
    # SUMMARY REPORT CARD
    # -------------------------------
    st.subheader("📝 Health Summary Report")
    report = pd.DataFrame({
        "Feature": [
            "Age", "Sleep Duration", "Quality of Sleep",
            "Physical Activity", "Stress Level",
            "Heart Rate", "Daily Steps",
            "BP Systolic", "BP Diastolic"
        ],
        "Value": [
            age, sleep_duration, quality_of_sleep,
            physical_activity_level, stress_level,
            heart_rate, daily_steps,
            bp_systolic, bp_diastolic
        ]
    })
    st.dataframe(report, use_container_width=True)

# -------------------------------
# FOOTER
# -------------------------------
st.markdown("---")
st.caption("⚠️ This app is for educational purposes only. Consult a doctor for medical advice.")
