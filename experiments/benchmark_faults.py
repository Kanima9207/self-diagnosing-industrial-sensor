"""Benchmark residual statistics and CUSUM across the baseline scenarios."""

from pathlib import Path
import csv
import numpy as np

from src.cusum import cusum
from src.diagnostic_metrics import residual_metrics
from src.signal_generator import SCENARIOS, simulate_sensor


def run() -> None:
    t = np.arange(0.0, 120.0, 0.1)
    rows = []

    for name, scenario in SCENARIOS.items():
        truth, measured = simulate_sensor(t, scenario)
        residual = measured - truth
        metrics = residual_metrics(measured, truth)
        _, alarms = cusum(residual, reference_mean=0.0, drift=0.05, threshold=2.0)
        metrics.update(
            scenario=name,
            cusum_alarm_rate=float(np.mean(alarms)),
        )
        rows.append(metrics)

    output = Path("results/diagnostic_metrics.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["scenario", "residual_mean", "residual_std", "residual_rms", "max_abs_residual", "cusum_alarm_rate"])
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(
            f"{row['scenario']:8s} | mean={row['residual_mean']:+.3f} | "
            f"std={row['residual_std']:.3f} | rms={row['residual_rms']:.3f} | "
            f"CUSUM alarms={row['cusum_alarm_rate']:.1%}"
        )


if __name__ == "__main__":
    run()
