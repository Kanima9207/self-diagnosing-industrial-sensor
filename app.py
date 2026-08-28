"""Interactive Streamlit dashboard for the self-diagnosing sensor."""

from __future__ import annotations

import numpy as np
import streamlit as st

from src.fault_classifier import classify_features
from src.feature_extraction import extract_features
from src.signal_generator import SensorScenario, simulate_sensor, true_process

st.set_page_config(page_title="Industrial Sensor Health Monitor", page_icon="🔧", layout="wide")

st.title("🔧 Industrial Sensor Health Monitor")
st.caption("Synthetic sensor-fault detection using signal processing, CUSUM, and interpretable classification.")

with st.sidebar:
    st.header("Simulation controls")
    fault = st.selectbox("Fault mode", ["healthy", "noisy", "bias", "drift", "stuck"])
    severity = st.slider("Fault severity", 0.5, 3.0, 1.0, 0.1)
    noise = st.slider("Noise standard deviation", 0.05, 1.50, 0.15, 0.05)
    seed = st.number_input("Random seed", min_value=0, max_value=99999, value=2026, step=1)

# Map a single severity control to physically meaningful scenario parameters.
if fault == "healthy":
    scenario = SensorScenario(fault, noise_std=noise)
elif fault == "noisy":
    scenario = SensorScenario(fault, noise_std=max(noise, 0.45 * severity))
elif fault == "bias":
    scenario = SensorScenario(fault, noise_std=noise, bias=severity)
elif fault == "drift":
    scenario = SensorScenario(fault, noise_std=noise, drift_rate=0.01 * severity)
else:
    scenario = SensorScenario(fault, noise_std=noise, stuck_at=25.0 + severity)

t = np.arange(0.0, 60.0, 0.1)
truth, measured = simulate_sensor(t, scenario, seed=int(seed), fault_start_time=20.0 if fault != "healthy" else None)
residual = measured - truth
window = residual[int(len(t) / 3):]
measured_window = measured[int(len(t) / 3):]
truth_window = truth[int(len(t) / 3):]
features = extract_features(window, measured=measured_window, reference=truth_window)
predicted, confidence = classify_features(features)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Diagnosis", predicted.upper())
col2.metric("Confidence", f"{confidence:.1%}")
col3.metric("Residual RMS", f"{features['rms']:.3f}")
col4.metric("Sensor-process correlation", f"{features['process_sensor_correlation']:.3f}")

st.subheader("Sensor signal")
chart_data = {"True process": truth, "Measured sensor": measured}
st.line_chart(chart_data)

st.subheader("Residual")
st.line_chart({"Residual": residual})

st.subheader("Extracted diagnostic features")
st.dataframe({k: [round(v, 5)] for k, v in features.items()}, use_container_width=True)

st.info("This dashboard operates on synthetic data. It is intended as an engineering demonstration, not a safety-critical diagnostic system.")
