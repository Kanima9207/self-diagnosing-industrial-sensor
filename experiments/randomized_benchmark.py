"""Repeated randomized benchmark with controlled fault onset."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from src.cusum import cusum
from src.diagnostic_metrics import detection_delay, false_alarm_rate
from src.signal_generator import SensorScenario, simulate_sensor

FAULTS = ("healthy", "noisy", "bias", "drift", "stuck")


def make_scenario(fault: str, rng: np.random.Generator) -> SensorScenario:
    """Create one randomized fault instance."""
    base_noise = float(rng.uniform(0.08, 0.20))
    if fault == "healthy":
        return SensorScenario(fault, noise_std=base_noise)
    if fault == "noisy":
        return SensorScenario(fault, noise_std=float(rng.uniform(0.55, 1.10)))
    if fault == "bias":
        return SensorScenario(fault, noise_std=base_noise, bias=float(rng.uniform(1.2, 3.0)))
    if fault == "drift":
        return SensorScenario(fault, noise_std=base_noise, drift_rate=float(rng.uniform(0.015, 0.045)))
    if fault == "stuck":
        return SensorScenario(fault, noise_std=base_noise, stuck_at=float(rng.uniform(23.0, 27.0)))
    raise ValueError(f"Unknown fault: {fault}")


def run_trials(trials: int = 100, seed: int = 2026) -> Path:
    """Run 100 randomized trials per fault with onset at 60 seconds."""
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, 120.0, 0.1)
    sampling_period = 0.1
    fault_start_time = 60.0
    fault_start_index = int(fault_start_time / sampling_period)
    rows: list[dict[str, object]] = []

    for fault in FAULTS:
        for trial in range(1, trials + 1):
            scenario = make_scenario(fault, rng)
            truth, measured = simulate_sensor(
                t,
                scenario,
                seed=int(rng.integers(0, 2**32 - 1)),
                fault_start_time=None if fault == "healthy" else fault_start_time,
            )
            residual = measured - truth
            _, alarms = cusum(residual, reference_mean=0.0, drift=0.05, threshold=2.0)

            if fault == "healthy":
                far = float(np.mean(alarms))
                delay = None
                post_alarm_rate = float(np.mean(alarms))
            else:
                far = false_alarm_rate(alarms, fault_start_index)
                delay = detection_delay(alarms, fault_start_index, sampling_period)
                post_alarm_rate = float(np.mean(alarms[fault_start_index:]))

            detected = delay is not None
            rows.append(
                {
                    "fault": fault,
                    "trial": trial,
                    "noise_std": scenario.noise_std,
                    "bias": scenario.bias,
                    "drift_rate": scenario.drift_rate,
                    "stuck_at": "" if scenario.stuck_at is None else scenario.stuck_at,
                    "cusum_alarm_rate": post_alarm_rate,
                    "false_alarm_rate": far,
                    "detected": int(detected),
                    "detection_delay_s": "" if delay is None else delay,
                }
            )

    output = Path("results/randomized_benchmark.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output


if __name__ == "__main__":
    path = run_trials()
    print(f"Saved randomized benchmark to {path}")
    print("Trials: 100 per fault class (500 total)")
