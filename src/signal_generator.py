"""Synthetic industrial process and sensor-fault generator."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class SensorScenario:
    """Configuration for a sensor condition."""

    name: str
    noise_std: float = 0.15
    bias: float = 0.0
    drift_rate: float = 0.0
    stuck_at: float | None = None


def true_process(t: np.ndarray) -> np.ndarray:
    """Return a slowly varying process variable."""
    return 25.0 + 2.5 * np.sin(2 * np.pi * 0.05 * t) + 0.5 * np.sin(2 * np.pi * 0.17 * t)


def simulate_sensor(
    t: np.ndarray,
    scenario: SensorScenario,
    seed: int = 42,
    fault_start_time: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a measurement with an optional fault onset.

    Before ``fault_start_time`` the sensor is healthy. At and after the onset,
    the configured fault is applied. This allows genuine detection-delay and
    false-alarm measurements.
    """
    rng = np.random.default_rng(seed)
    t = np.asarray(t, dtype=float)
    truth = true_process(t)
    measured = truth + rng.normal(0.0, scenario.noise_std, size=t.shape)

    if fault_start_time is None:
        active = np.ones_like(t, dtype=bool)
    else:
        active = t >= fault_start_time

    if scenario.bias != 0.0:
        measured[active] += scenario.bias

    if scenario.drift_rate != 0.0 and np.any(active):
        measured[active] += scenario.drift_rate * (t[active] - fault_start_time)

    if scenario.stuck_at is not None:
        measured[active] = scenario.stuck_at

    return truth, measured


SCENARIOS = {
    "healthy": SensorScenario("healthy"),
    "noisy": SensorScenario("noisy", noise_std=0.8),
    "bias": SensorScenario("bias", bias=2.0),
    "drift": SensorScenario("drift", drift_rate=0.025),
    "stuck": SensorScenario("stuck", stuck_at=25.0),
}
