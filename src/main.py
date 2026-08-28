"""Run the Phase-1 sensor simulation and save diagnostic plots."""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from .fault_detection import classify_fault
from .signal_generator import SCENARIOS, simulate_sensor
from .signal_processing import moving_mean, normalized_error


def run(output_dir: str = "results") -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    t = np.arange(0.0, 120.0, 0.1)
    for name, scenario in SCENARIOS.items():
        truth, measured = simulate_sensor(t, scenario)
        filtered = moving_mean(measured, window=25)
        label = classify_fault(measured, truth)
        error = normalized_error(measured, truth)

        fig, ax = plt.subplots(figsize=(10, 4.8))
        ax.plot(t, truth, label="True process", linewidth=2)
        ax.plot(t, measured, label="Measured", alpha=0.55)
        ax.plot(t, filtered, label="Moving mean", linewidth=2)
        ax.set_title(f"{name.title()} sensor | baseline diagnosis: {label}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Process variable")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"{name}.png", dpi=160)
        plt.close(fig)
        print(f"{name:8s} | diagnosis={label:7s} | normalized RMS error={error:.4f}")


if __name__ == "__main__":
    run()
