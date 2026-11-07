"""Small input checks shared by the public functions."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def as_1d(name: str, values: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite numbers")
    return arr


def as_pd(name: str, values: ArrayLike, *, open_interval: bool = True) -> NDArray[np.float64]:
    arr = as_1d(name, values)
    if open_interval:
        if np.any((arr <= 0.0) | (arr >= 1.0)):
            raise ValueError(f"{name} must lie strictly between 0 and 1")
    elif np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError(f"{name} must lie between 0 and 1")
    return arr


def as_counts(name: str, values: ArrayLike) -> NDArray[np.float64]:
    arr = as_1d(name, values)
    if np.any(arr < 0) or np.any(arr != np.round(arr)):
        raise ValueError(f"{name} must be non-negative whole numbers")
    return arr


def obligors_and_defaults(
    n_obligors: ArrayLike, n_defaults: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    n = as_counts("n_obligors", n_obligors)
    d = as_counts("n_defaults", n_defaults)
    if n.shape != d.shape:
        raise ValueError("n_obligors and n_defaults must have the same length")
    if np.any(d > n):
        raise ValueError("n_defaults cannot exceed n_obligors")
    return n, d


def probability(name: str, value: float) -> float:
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must lie strictly between 0 and 1")
    return float(value)
