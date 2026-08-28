"""Synthetic industrial process and sensor-fault generator."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class SensorScenario:
    """Configuration for one simulated sensor condition."""

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
) -> tuple[np.ndarray, np.ndarray]:
    """Generate true and measured signals for a scenario.

    The model is y(t) = x(t) + n(t) + b(t) + d(t).
    A stuck sensor overrides the measurement with a constant value.
    """
    rng = np.random.default_rng(seed)
    truth = true_process(t)
    noise = rng.normal(0.0, scenario.noise_std, size=t.shape)
    bias = np.full_like(t, scenario.bias, dtype=float)
    drift = scenario.drift_rate * np.maximum(t - t[0], 0.0)
    measured = truth + noise + bias + drift

    if scenario.stuck_at is not None:
        measured = np.full_like(t, scenario.stuck_at, dtype=float)

    return truth, measured


SCENARIOS = {
    "healthy": SensorScenario("healthy"),
    "noisy": SensorScenario("noisy", noise_std=0.8),
    "bias": SensorScenario("bias", bias=2.0),
    "drift": SensorScenario("drift", drift_rate=0.025),
    "stuck": SensorScenario("stuck", stuck_at=25.0),
}
