"""Gradual sensor-drift experiment."""
from src.signal_generator import SCENARIOS, simulate_sensor
import numpy as np

if __name__ == "__main__":
    t = np.arange(0, 120, 0.1)
    truth, measured = simulate_sensor(t, SCENARIOS["drift"])
    drift = measured[-1] - truth[-1]
    print(f"final injected drift={drift:.3f}")
