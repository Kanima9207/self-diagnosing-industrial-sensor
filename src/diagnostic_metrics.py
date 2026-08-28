"""Quantitative metrics for sensor-diagnostic experiments."""

from __future__ import annotations

import numpy as np
from .signal_processing import rms


def residual_metrics(measured: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Return interpretable statistics of measurement residuals."""
    residual = np.asarray(measured, dtype=float) - np.asarray(reference, dtype=float)
    return {
        "residual_mean": float(np.mean(residual)),
        "residual_std": float(np.std(residual)),
        "residual_rms": rms(residual),
        "max_abs_residual": float(np.max(np.abs(residual))),
    }


def detection_delay(
    fault_flags: np.ndarray,
    fault_start_index: int,
    sampling_period: float,
    persistence: int = 5,
) -> float | None:
    """Return seconds from fault onset to the first persistent detection."""
    flags = np.asarray(fault_flags, dtype=bool)
    for i in range(max(fault_start_index, 0), len(flags) - persistence + 1):
        if np.all(flags[i : i + persistence]):
            return float((i - fault_start_index) * sampling_period)
    return None


def false_alarm_rate(fault_flags: np.ndarray, healthy_end_index: int) -> float:
    """Fraction of healthy samples incorrectly flagged."""
    flags = np.asarray(fault_flags, dtype=bool)
    end = min(max(healthy_end_index, 0), len(flags))
    if end == 0:
        return 0.0
    return float(np.mean(flags[:end]))
