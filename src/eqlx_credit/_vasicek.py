"""The one-factor Vasicek (asymptotic single risk factor) model, shared by the tests that allow for
correlated defaults.

Given a systematic factor Z ~ N(0, 1) and asset correlation rho, an obligor with unconditional PD p defaults
with conditional probability ``Phi((Phi^-1(p) - sqrt(rho) Z) / sqrt(1 - rho))``. Defaults are independent
given Z, so the number of defaults among n obligors is a binomial mixture over Z.
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def conditional_pd(pd: float, rho: float, z: np.ndarray) -> np.ndarray:
    return np.asarray(
        stats.norm.cdf((stats.norm.ppf(pd) - np.sqrt(rho) * z) / np.sqrt(1.0 - rho)), dtype=float
    )


def vasicek_cdf(defaults: float, obligors: float, pd: float, rho: float, nodes: int = 96) -> float:
    """P(X <= defaults) under the one-factor model, integrated over Z by Gauss-Hermite quadrature."""
    if defaults < 0:
        return 0.0
    x, w = np.polynomial.hermite_e.hermegauss(nodes)
    w = w / np.sqrt(2.0 * np.pi)
    return float(np.dot(w, stats.binom.cdf(defaults, obligors, conditional_pd(pd, rho, x))))
