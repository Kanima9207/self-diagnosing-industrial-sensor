"""Interpretable features used for sensor-fault classification."""

from __future__ import annotations

import numpy as np

from .signal_processing import rms


def extract_features(residual: np.ndarray, sampling_period: float = 0.1) -> dict[str, float]:
    """Extract level, variability, energy, and trend features from residuals."""
    r = np.asarray(residual, dtype=float)
    if r.size < 2:
        raise ValueError("At least two residual samples are required")
    time = np.arange(r.size, dtype=float) * sampling_period
    slope = float(np.polyfit(time, r, 1)[0])
    return {
        "mean": float(np.mean(r)),
        "abs_mean": float(np.mean(np.abs(r))),
        "std": float(np.std(r)),
        "rms": rms(r),
        "peak": float(np.max(np.abs(r))),
        "slope": slope,
    }
