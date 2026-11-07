"""PD estimation for low-default portfolios (LDP).

Some portfolios (large corporates, banks, sovereigns, project finance) see so few
defaults that grade-level default rates are zero or near zero, yet the model still
needs a positive, ordered PD per grade. Two published methods:

* :func:`pluto_tasche`: the "most prudent estimation" of Pluto and Tasche (2005).
  An upper confidence bound per grade, pooling the grade with all riskier grades.
* :func:`van_der_burgt`: van der Burgt (2008). Fits a one-parameter concave CAP curve
  to a target accuracy ratio and reads grade PDs off its slope, scaled to the
  central tendency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats
from scipy.optimize import brentq

from ._checks import as_1d, obligors_and_defaults, probability
from ._vasicek import vasicek_cdf


def _pooled_counts(
    n: NDArray[np.float64], d: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """For each grade, obligors and defaults in that grade plus all riskier grades."""
    return np.cumsum(n[::-1])[::-1], np.cumsum(d[::-1])[::-1]


def _vasicek_cdf(defaults: float, obligors: float, pd: float, rho: float, nodes: int) -> float:
    """P(X <= defaults) when defaults follow the one-factor Vasicek model."""
    return vasicek_cdf(defaults, obligors, pd, rho, nodes)


def pluto_tasche(
    n_obligors: ArrayLike,
    n_defaults: ArrayLike,
    confidence: float = 0.9,
    rho: float = 0.0,
    quadrature_nodes: int = 96,
) -> NDArray[np.float64]:
    """Most prudent PD upper bounds per grade (Pluto and Tasche, 2005).

    Grades are ordered from lowest to highest risk. For grade ``i`` the bound is the
    largest PD ``p`` that, if every grade from ``i`` to the riskiest had PD ``p``,
    would still make the observed defaults (or fewer) at least ``1 - confidence``
    likely. The bounds are therefore non-decreasing and positive even with zero
    defaults.

    ``rho = 0`` assumes independent defaults (closed form via the Beta distribution);
    ``rho > 0`` uses the one-factor Vasicek model with asset correlation ``rho``, which
    gives higher, more prudent bounds. Single-period version.
    """
    n, d = obligors_and_defaults(n_obligors, n_defaults)
    if np.any(n == 0):
        raise ValueError("every grade needs at least one obligor")
    gamma = probability("confidence", confidence)
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must lie in [0, 1)")
    big_n, big_d = _pooled_counts(n, d)

    target = 1.0 - gamma
    bounds = np.empty_like(n)
    for i, (obl, dft) in enumerate(zip(big_n, big_d, strict=True)):
        if dft >= obl:
            bounds[i] = 1.0
        elif rho == 0.0:
            bounds[i] = stats.beta.ppf(gamma, dft + 1.0, obl - dft)
        else:
            bounds[i] = brentq(
                lambda p, o=obl, k=dft: _vasicek_cdf(k, o, p, rho, quadrature_nodes) - target,
                1e-12,
                1.0 - 1e-12,
                xtol=1e-14,
            )
    return bounds


def _cap(x: NDArray[np.float64] | float, k: float) -> NDArray[np.float64]:
    """van der Burgt CAP curve C(x) = (1 - e^{-kx}) / (1 - e^{-k})."""
    return np.asarray(-np.expm1(-k * np.asarray(x)) / -np.expm1(-k), dtype=float)


def _vdb_accuracy_ratio(k: float, central_tendency: float) -> float:
    area = 1.0 / -np.expm1(-k) - 1.0 / k
    return float((2.0 * area - 1.0) / (1.0 - central_tendency))


@dataclass(frozen=True)
class VanDerBurgtFit:
    pd: NDArray[np.float64]
    concavity: float
    central_tendency: float
    accuracy_ratio: float


def van_der_burgt(weights: ArrayLike, central_tendency: float, accuracy_ratio: float) -> VanDerBurgtFit:
    """Grade PDs from a fitted CAP curve (van der Burgt, 2008).

    ``weights`` is the number (or share) of obligors per grade, lowest risk first.
    The CAP curve ``C(x) = (1 - e^{-kx}) / (1 - e^{-k})`` runs over obligors sorted
    from riskiest to safest; ``k`` is chosen so the curve's accuracy ratio equals
    ``accuracy_ratio``. A grade's PD is the central tendency times the average slope
    of the curve over that grade's obligors, so the weighted average PD equals the
    central tendency exactly.
    """
    w = as_1d("weights", weights)
    if w.size < 2 or np.any(w <= 0):
        raise ValueError("need at least two grades, each with positive weight")
    w = w / w.sum()
    ct = probability("central_tendency", central_tendency)
    target = probability("accuracy_ratio", accuracy_ratio)

    k = brentq(lambda kk: _vdb_accuracy_ratio(kk, ct) - target, 1e-8, 700.0, xtol=1e-12)
    riskier_or_equal = np.cumsum(w[::-1])[::-1]
    riskier = riskier_or_equal - w
    pd = ct * (_cap(riskier_or_equal, k) - _cap(riskier, k)) / w
    if np.any(pd >= 1.0):
        raise ValueError("fitted PDs reach 1; the target accuracy ratio is too high")
    return VanDerBurgtFit(pd=pd, concavity=float(k), central_tendency=ct, accuracy_ratio=target)
