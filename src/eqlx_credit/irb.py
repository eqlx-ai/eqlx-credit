"""Basel IRB risk-weight functions for credit risk.

The capital requirement K per unit of exposure, from PD, LGD and (for corporates) maturity:

    K = [LGD * N((G(PD) + sqrt(R) * G(0.999)) / sqrt(1 - R)) - PD * LGD] * MA

with N the standard normal CDF, G its inverse, R the asset correlation of the asset class and MA the
maturity adjustment ``(1 + (M - 2.5) b) / (1 - 1.5 b)``, ``b = (0.11852 - 0.05478 ln PD)^2`` (corporate,
sovereign and bank exposures only). The risk weight is ``K * 12.5 * scaling``.

Sources: Basel Committee, *International Convergence of Capital Measurement and Capital Standards*
(June 2006), paragraphs 272-273 and 328-330, and the BCBS *Explanatory Note on the Basel II IRB Risk
Weight Functions* (2005). The tests reproduce the illustrative corporate risk weights of that document's
Annex 5. ``scaling=1.06`` gives the Basel II / CRR scaling factor; Basel III final (CRE31) dropped it.
PD floors differ by framework (Basel II and CRR: 0.03%; Basel III final: 0.05% for most classes) and are
the caller's choice.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from ._checks import as_1d

AssetClass = Literal["corporate", "residential_mortgage", "qrre", "other_retail"]


def _interp(pd: NDArray[np.float64], k: float, lo: float, hi: float) -> NDArray[np.float64]:
    f = -np.expm1(-k * pd) / -np.expm1(-k)
    return np.asarray(lo * f + hi * (1.0 - f), dtype=float)


def asset_correlation(
    pd: ArrayLike, asset_class: AssetClass = "corporate", sales_eur_m: ArrayLike | None = None
) -> NDArray[np.float64]:
    """Asset correlation R. ``sales_eur_m`` (annual sales in EUR million, clamped to 5..50) applies the
    SME firm-size adjustment ``-0.04 * (1 - (S - 5) / 45)`` to corporates with sales below 50."""
    p = as_1d("pd", np.atleast_1d(np.asarray(pd, dtype=float)))
    if asset_class == "corporate":
        r = _interp(p, 50.0, 0.12, 0.24)
        if sales_eur_m is not None:
            s = np.clip(np.broadcast_to(np.asarray(sales_eur_m, dtype=float), p.shape), 5.0, 50.0)
            r = r - 0.04 * (1.0 - (s - 5.0) / 45.0)
        return r
    if sales_eur_m is not None:
        raise ValueError("the firm-size adjustment applies to corporate exposures only")
    if asset_class == "residential_mortgage":
        return np.full_like(p, 0.15)
    if asset_class == "qrre":
        return np.full_like(p, 0.04)
    if asset_class == "other_retail":
        return _interp(p, 35.0, 0.03, 0.16)
    raise ValueError(f"unknown asset class {asset_class!r}")


def capital_requirement(
    pd: ArrayLike,
    lgd: ArrayLike,
    asset_class: AssetClass = "corporate",
    maturity: ArrayLike = 2.5,
    sales_eur_m: ArrayLike | None = None,
    pd_floor: float = 0.0,
) -> NDArray[np.float64]:
    """Capital requirement K per unit of exposure (a share of EAD). ``maturity`` in years (1..5) applies to
    corporates only. PDs below ``pd_floor`` are raised to it; PD must be below 1 (a defaulted exposure's
    K is ``max(0, LGD - expected loss best estimate)`` and is not computed here)."""
    p = np.maximum(as_1d("pd", np.atleast_1d(np.asarray(pd, dtype=float))), pd_floor)
    if np.any((p <= 0) | (p >= 1)):
        raise ValueError("pd must lie strictly between 0 and 1 (after the floor)")
    lgd_a = np.broadcast_to(np.asarray(lgd, dtype=float), p.shape)
    if np.any((lgd_a < 0) | (lgd_a > 1)):
        raise ValueError("lgd must lie between 0 and 1")
    r = asset_correlation(p, asset_class, sales_eur_m)
    g = stats.norm.ppf(p)
    k = lgd_a * stats.norm.cdf((g + np.sqrt(r) * stats.norm.ppf(0.999)) / np.sqrt(1.0 - r)) - p * lgd_a
    if asset_class == "corporate":
        m = np.clip(np.broadcast_to(np.asarray(maturity, dtype=float), p.shape), 1.0, 5.0)
        b = (0.11852 - 0.05478 * np.log(p)) ** 2
        k = k * (1.0 + (m - 2.5) * b) / (1.0 - 1.5 * b)
    return np.asarray(k, dtype=float)


def risk_weight(
    pd: ArrayLike,
    lgd: ArrayLike,
    asset_class: AssetClass = "corporate",
    maturity: ArrayLike = 2.5,
    sales_eur_m: ArrayLike | None = None,
    pd_floor: float = 0.0,
    scaling: float = 1.0,
) -> NDArray[np.float64]:
    """Risk weight as a share (0.9232 means 92.32%): ``K * 12.5 * scaling``. Multiply by EAD for RWA."""
    k = capital_requirement(pd, lgd, asset_class, maturity, sales_eur_m, pd_floor)
    return k * 12.5 * scaling
