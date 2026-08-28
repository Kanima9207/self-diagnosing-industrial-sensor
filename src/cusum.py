"""CUSUM change-detection implementation."""

from __future__ import annotations

import numpy as np


def cusum(
    signal: np.ndarray,
    reference_mean: float | None = None,
    drift: float = 0.05,
    threshold: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate two-sided CUSUM statistics and alarm flags.

    Parameters are intentionally explicit so experiments can study the
    sensitivity/specificity trade-off rather than hiding it in a model.
    """
    x = np.asarray(signal, dtype=float)
    mu = float(np.mean(x)) if reference_mean is None else float(reference_mean)
    positive = np.zeros_like(x)
    negative = np.zeros_like(x)
    alarms = np.zeros_like(x, dtype=bool)

    for i, value in enumerate(x):
        previous_pos = positive[i - 1] if i else 0.0
        previous_neg = negative[i - 1] if i else 0.0
        positive[i] = max(0.0, previous_pos + value - mu - drift)
        negative[i] = max(0.0, previous_neg + mu - value - drift)
        alarms[i] = positive[i] > threshold or negative[i] > threshold

    statistic = np.maximum(positive, negative)
    return statistic, alarms
