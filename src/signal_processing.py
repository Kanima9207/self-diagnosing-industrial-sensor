"""Signal-processing utilities used by the diagnostic pipeline."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d


def moving_mean(signal: np.ndarray, window: int = 25) -> np.ndarray:
    """Calculate a centered moving mean."""
    if window < 2:
        raise ValueError("window must be at least 2")
    return uniform_filter1d(np.asarray(signal, dtype=float), size=window, mode="nearest")


def moving_std(signal: np.ndarray, window: int = 25) -> np.ndarray:
    """Calculate a centered moving standard deviation."""
    x = np.asarray(signal, dtype=float)
    mean = moving_mean(x, window)
    mean_sq = moving_mean(x * x, window)
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))


def residual(measured: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return measurement residual relative to a reference estimate."""
    return np.asarray(measured, dtype=float) - np.asarray(reference, dtype=float)


def rms(signal: np.ndarray) -> float:
    """Return root-mean-square magnitude."""
    x = np.asarray(signal, dtype=float)
    return float(np.sqrt(np.mean(x * x)))


def normalized_error(measured: np.ndarray, reference: np.ndarray) -> float:
    """Return RMS residual normalized by RMS reference magnitude."""
    ref_rms = rms(reference)
    if ref_rms == 0:
        return float("inf")
    return rms(residual(measured, reference)) / ref_rms
