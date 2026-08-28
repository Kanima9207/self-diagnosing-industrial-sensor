"""Interpretable features used for sensor-fault classification."""

from __future__ import annotations

import numpy as np

from .signal_processing import rms


def extract_features(
    residual: np.ndarray,
    sampling_period: float = 0.1,
    measured: np.ndarray | None = None,
    reference: np.ndarray | None = None,
) -> dict[str, float]:
    """Extract level, variability, energy, trend, and response features.

    ``measured`` and ``reference`` are optional so existing callers remain
    compatible. Correlation and variance-ratio features help distinguish a
    sensor that has stopped responding from one that has a simple offset.
    """
    r = np.asarray(residual, dtype=float)
    if r.size < 2:
        raise ValueError("At least two residual samples are required")
    time = np.arange(r.size, dtype=float) * sampling_period
    slope = float(np.polyfit(time, r, 1)[0])
    features = {
        "mean": float(np.mean(r)),
        "abs_mean": float(np.mean(np.abs(r))),
        "std": float(np.std(r)),
        "rms": rms(r),
        "peak": float(np.max(np.abs(r))),
        "slope": slope,
    }
    if measured is not None and reference is not None:
        y = np.asarray(measured, dtype=float)
        x = np.asarray(reference, dtype=float)
        if y.shape != x.shape or y.shape != r.shape:
            raise ValueError("measured, reference, and residual must have the same shape")
        y_std = float(np.std(y))
        x_std = float(np.std(x))
        if y_std == 0.0 or x_std == 0.0:
            correlation = 0.0
        else:
            correlation = float(np.corrcoef(x, y)[0, 1])
        features["process_sensor_correlation"] = correlation
        features["variance_ratio"] = float((y_std**2) / max(x_std**2, 1e-12))
    return features
