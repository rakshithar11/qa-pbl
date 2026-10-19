import os
import cv2
import joblib
import numpy as np
import streamlit as st

from thread_features import extract_thread_features

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "thread_pitch_classifier.pkl")


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


st.title("🔩 Optical Screw Thread Measurement System")
st.write("Automatic pitch-based calibration and thread metrology")

model = load_model()
if model is None:
    st.error("Model not found. Run `python code/train_model.py` first.")
    st.stop()

uploaded = st.file_uploader("Upload thread silhouette image", type=["png", "jpg", "jpeg"])

if uploaded:
    img = cv2.imdecode(np.frombuffer(uploaded.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        st.error("Could not read image.")
        st.stop()

    st.image(img, caption="Uploaded Image", channels="BGR")
    result = extract_thread_features(img)

    if result is None:
        st.error("❌ Thread detection failed")
    else:
        X = np.array(result["features"]).reshape(1, -1)
        pitch_mm = float(model.predict(X)[0])      # labels are pitch in mm
        scale = pitch_mm / result["pitch_px"]       # mm per px

        # Measured diameters from the silhouette envelope
        height_px = result["y_bottom"] - result["y_top"]
        major_mm = np.percentile(height_px, 95) * scale
        minor_mm_measured = np.percentile(height_px, 5) * scale
        minor_mm_theory = major_mm - 2 * 0.6134 * pitch_mm  # ISO metric external thread

        draw = img.copy()
        for i in result["crests"]:
            cv2.circle(draw, (int(result["x"][i]), int(result["y_top"][i])), 4, (255, 0, 0), -1)
        for i in result["roots"]:
            cv2.circle(draw, (int(result["x"][i]), int(result["y_bottom"][i])), 4, (0, 0, 255), -1)
        st.image(draw, caption="Crests (blue) and roots (red)", channels="BGR")

        st.success(f"Predicted Pitch: {pitch_mm:.2f} mm")
        st.write(f"Auto-calibrated scale: {scale:.5f} mm/px")
        st.write(f"Major Diameter: {major_mm:.2f} mm")
        st.write(f"Minor Diameter (measured): {minor_mm_measured:.2f} mm")
        st.write(f"Minor Diameter (ISO theoretical): {minor_mm_theory:.2f} mm")
        st.write("Thread Angle: 60° (ISO Metric)")
