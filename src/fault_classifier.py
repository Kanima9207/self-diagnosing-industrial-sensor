"""Transparent rule-based sensor-fault classifier."""

from __future__ import annotations

from typing import Mapping


def classify_features(features: Mapping[str, float]) -> tuple[str, float]:
    """Classify residual behavior and return (label, confidence).

    This is an intentionally interpretable baseline. Confidence is a heuristic
    score, not a calibrated probability.
    """
    mean = abs(features["mean"])
    std = features["std"]
    rms = features["rms"]
    peak = features["peak"]
    slope = abs(features["slope"])

    if std < 0.02 and peak > 0.5:
        return "stuck", min(0.99, 0.80 + peak / 100.0)
    if mean > 1.0 and mean > 2.0 * std:
        return "bias", min(0.99, 0.70 + mean / 10.0)
    if slope > 0.01 and mean < 1.5:
        return "drift", min(0.99, 0.70 + slope * 5.0)
    if std > 0.45 or rms > 0.5:
        return "noisy", min(0.99, 0.65 + std / 3.0)
    return "healthy", max(0.50, min(0.99, 1.0 - min(0.5, rms)))
