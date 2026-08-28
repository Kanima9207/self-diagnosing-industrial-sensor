"""Interpretable statistical fault-detection methods."""

from __future__ import annotations

import numpy as np
from .signal_processing import moving_mean, moving_std


def detect_stuck(signal: np.ndarray, window: int = 25, std_threshold: float = 1e-3) -> np.ndarray:
    """Flag windows whose local variation is effectively zero."""
    return moving_std(signal, window) < std_threshold


def detect_bias(
    residual: np.ndarray,
    window: int = 25,
    threshold: float = 1.0,
) -> np.ndarray:
    """Flag sustained residual offsets above a threshold."""
    local_bias = np.abs(moving_mean(residual, window))
    return local_bias > threshold


def detect_noise(
    signal: np.ndarray,
    window: int = 25,
    std_threshold: float = 0.5,
) -> np.ndarray:
    """Flag locally elevated measurement noise."""
    return moving_std(signal, window) > std_threshold


def detect_drift(
    residual: np.ndarray,
    window: int = 51,
    slope_threshold: float = 0.01,
) -> np.ndarray:
    """Estimate local residual slope and flag sustained drift."""
    x = np.asarray(residual, dtype=float)
    half = window // 2
    flags = np.zeros(x.size, dtype=bool)
    for i in range(half, x.size - half):
        segment = x[i - half : i + half + 1]
        time = np.arange(segment.size, dtype=float)
        slope = np.polyfit(time, segment, 1)[0]
        flags[i] = abs(slope) > slope_threshold
    return flags


def classify_fault(
    measured: np.ndarray,
    reference: np.ndarray,
    window: int = 25,
) -> str:
    """Return a simple baseline fault label using interpretable rules."""
    residual = np.asarray(measured) - np.asarray(reference)
    if np.mean(detect_stuck(measured, window)) > 0.5:
        return "stuck"
    if np.mean(detect_bias(residual, window)) > 0.5:
        return "bias"
    if np.mean(detect_drift(residual)) > 0.2:
        return "drift"
    if np.mean(detect_noise(residual, window, std_threshold=0.5)) > 0.2:
        return "noisy"
    return "healthy"
