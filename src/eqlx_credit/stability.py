"""Stability and concentration: has the portfolio moved across grades, and is it bunched in a few?"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import ArrayLike

from ._checks import as_1d, as_counts


def psi(expected: ArrayLike, actual: ArrayLike, epsilon: float = 1e-6) -> float:
    """Population stability index between two distributions over the same grades.

    Accepts counts or shares. A grade empty in either distribution is floored at ``epsilon`` so the log is
    defined; that floor inflates the index, so a warning is raised. A widely used reading (Siddiqi, 2006,
    *Credit Risk Scorecards*): below 0.10 stable, 0.10 to 0.25 worth a look, above 0.25 a material shift.
    """
    e = as_1d("expected", expected)
    a = as_1d("actual", actual)
    if e.shape != a.shape:
        raise ValueError("expected and actual must have the same length")
    if np.any(e < 0) or np.any(a < 0) or e.sum() == 0 or a.sum() == 0:
        raise ValueError("expected and actual must be non-negative and not all zero")
    e, a = e / e.sum(), a / a.sum()
    if np.any(e == 0) or np.any(a == 0):
        warnings.warn("an empty grade was floored at epsilon; the PSI is inflated", stacklevel=2)
    e, a = np.maximum(e, epsilon), np.maximum(a, epsilon)
    return float(np.sum((a - e) * np.log(a / e)))


def hhi(n_obligors: ArrayLike, normalized: bool = True, exposure: ArrayLike | None = None) -> float:
    """Herfindahl-Hirschman index of how the portfolio spreads across grades.

    Shares are obligor counts, or, with ``exposure`` (one total per grade), exposure amounts.
    ``normalized=True`` rescales to 0 (perfectly even) .. 1 (everything in one grade).
    """
    n = as_counts("n_obligors", n_obligors)
    if exposure is not None:
        x = as_1d("exposure", exposure)
        if x.shape != n.shape or np.any(x < 0):
            raise ValueError("exposure must be one non-negative amount per grade")
        n = x
    if n.sum() == 0:
        raise ValueError("the portfolio is empty")
    share = n / n.sum()
    value = float(np.sum(share**2))
    k = n.size
    if not normalized or k == 1:
        return value
    return (value - 1.0 / k) / (1.0 - 1.0 / k)
