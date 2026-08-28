"""Transparent rule-based sensor-fault classifier."""

from __future__ import annotations

from typing import Mapping


def classify_features(features: Mapping[str, float]) -> tuple[str, float]:
    """Classify residual behavior and return (label, confidence).

    Confidence is a heuristic score, not a calibrated probability.
    """
    mean = abs(features["mean"])
    std = features["std"]
    rms = features["rms"]
    peak = features["peak"]
    slope = abs(features["slope"])
    correlation = features.get("process_sensor_correlation")
    variance_ratio = features.get("variance_ratio")

    # A stuck sensor loses its normal response to process variation. Prefer
    # this physical signature over residual magnitude alone.
    if correlation is not None and variance_ratio is not None:
        if abs(correlation) < 0.20 and variance_ratio < 0.10:
            confidence = min(0.99, 0.80 + (0.20 - abs(correlation)) + (0.10 - variance_ratio))
            return "stuck", confidence

    if mean > 1.0 and mean > 2.0 * std:
        return "bias", min(0.99, 0.70 + mean / 10.0)
    if slope > 0.01 and mean < 1.5:
        return "drift", min(0.99, 0.70 + slope * 5.0)
    if std > 0.45 or rms > 0.5:
        return "noisy", min(0.99, 0.65 + std / 3.0)
    return "healthy", max(0.50, min(0.99, 1.0 - min(0.5, rms)))
