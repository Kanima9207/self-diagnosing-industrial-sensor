"""Stuck/frozen sensor experiment."""
from src.signal_generator import SCENARIOS, simulate_sensor
import numpy as np

if __name__ == "__main__":
    t = np.arange(0, 120, 0.1)
    _, measured = simulate_sensor(t, SCENARIOS["stuck"])
    print(f"unique measured values={np.unique(measured).size}")
