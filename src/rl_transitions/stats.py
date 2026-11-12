"""Confidence intervals."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Mean of `values` with a percentile bootstrap interval.

    Pass one value per prompt (for example a prompt's mean over its samples), so
    the resampling is over prompts and samples of the same prompt stay together.
    Returns (mean, low, high).
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size == 0:
        raise ValueError("need a non-empty 1-D array of values")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, x.size, size=(n_boot, x.size))].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(low), float(high)
