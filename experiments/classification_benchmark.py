"""Benchmark interpretable fault classification on post-onset windows."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.fault_classifier import classify_features
from src.feature_extraction import extract_features
from src.signal_generator import SensorScenario, simulate_sensor

FAULTS = ("healthy", "noisy", "bias", "drift", "stuck")


def make_scenario(fault: str, rng: np.random.Generator) -> SensorScenario:
    base = float(rng.uniform(0.08, 0.20))
    if fault == "healthy":
        return SensorScenario(fault, noise_std=base)
    if fault == "noisy":
        return SensorScenario(fault, noise_std=float(rng.uniform(0.55, 1.10)))
    if fault == "bias":
        return SensorScenario(fault, noise_std=base, bias=float(rng.uniform(1.2, 3.0)))
    if fault == "drift":
        return SensorScenario(fault, noise_std=base, drift_rate=float(rng.uniform(0.015, 0.045)))
    if fault == "stuck":
        return SensorScenario(fault, noise_std=base, stuck_at=float(rng.uniform(23.0, 27.0)))
    raise ValueError(fault)


def run(trials: int = 100, seed: int = 2026) -> Path:
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, 120.0, 0.1)
    onset = 60.0
    start = int(onset / 0.1)
    rows: list[dict[str, object]] = []

    for fault in FAULTS:
        for trial in range(1, trials + 1):
            scenario = make_scenario(fault, rng)
            truth, measured = simulate_sensor(t, scenario, seed=int(rng.integers(0, 2**32 - 1)), fault_start_time=None if fault == "healthy" else onset)
            # Use a fixed post-onset window. For healthy trials, use the same
            # final window length to avoid giving the classifier more context.
            residual = measured - truth
            window = residual[start:] if fault != "healthy" else residual[-len(residual[start:]):]
            features = extract_features(window)
            predicted, confidence = classify_features(features)
            rows.append({
                "fault": fault,
                "trial": trial,
                "predicted": predicted,
                "correct": int(predicted == fault),
                "confidence": confidence,
                **features,
            })

    output = Path("results/classification_benchmark.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output


if __name__ == "__main__":
    path = run()
    print(f"Saved classification benchmark to {path}")
    print("Trials: 100 per class (500 total)")
