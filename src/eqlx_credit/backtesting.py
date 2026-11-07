"""PD back-testing: are the assigned PDs consistent with the defaults that happened?

Per grade (one period): :func:`binomial_test` (optionally allowing for correlated defaults),
:func:`jeffreys_test`, and :func:`traffic_light`, which turns the correlated test into colours at
thresholds you choose. Across grades: :func:`hosmer_lemeshow`. Across years: :func:`normal_test`.
Obligor level: :func:`spiegelhalter`.

Every per-grade result carries the inputs (obligors, defaults, PD), the observed default rate and the
expected number of defaults, so it can go straight into a validation report table via ``rows()``.

Use the PD each obligor had at the start of the observation period, averaged per grade (obligor-weighted),
as the grade PD. A small p-value means more defaults than the PD implies: the PD looks too low.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from ._checks import as_1d, as_pd, obligors_and_defaults, probability
from ._vasicek import vasicek_cdf


@dataclass(frozen=True)
class GradeTest:
    """Per-grade result of a calibration test."""

    test: str
    n_obligors: NDArray[np.float64]
    n_defaults: NDArray[np.float64]
    pd: NDArray[np.float64]
    p_value: NDArray[np.float64]
    alpha: float

    @property
    def rejected(self) -> NDArray[np.bool_]:
        return np.asarray(self.p_value < self.alpha)

    @property
    def observed_rate(self) -> NDArray[np.float64]:
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.n_obligors > 0, self.n_defaults / self.n_obligors, np.nan)

    @property
    def expected_defaults(self) -> NDArray[np.float64]:
        return self.n_obligors * self.pd

    def rows(self) -> list[dict[str, Any]]:
        """One dict per grade, for a report table (grade numbers start at 1)."""
        return [
            {
                "grade": i + 1,
                "obligors": int(self.n_obligors[i]),
                "defaults": int(self.n_defaults[i]),
                "pd": float(self.pd[i]),
                "observed_rate": float(self.observed_rate[i]),
                "expected_defaults": float(self.expected_defaults[i]),
                "p_value": float(self.p_value[i]),
                "rejected": bool(self.rejected[i]),
            }
            for i in range(self.pd.size)
        ]


def _grade_inputs(
    n_obligors: ArrayLike, n_defaults: ArrayLike, pd: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    n, d = obligors_and_defaults(n_obligors, n_defaults)
    p = as_pd("pd", pd)
    if p.shape != n.shape:
        raise ValueError("pd must have one value per grade")
    return n, d, p


def _binomial_p(
    n: NDArray[np.float64], d: NDArray[np.float64], p: NDArray[np.float64], rho: float
) -> NDArray[np.float64]:
    if rho == 0.0:
        return np.asarray(stats.binom.sf(d - 1, n, p), dtype=float)
    return np.array([1.0 - vasicek_cdf(k - 1, m, q, rho) for m, k, q in zip(n, d, p, strict=True)])


def binomial_test(
    n_obligors: ArrayLike,
    n_defaults: ArrayLike,
    pd: ArrayLike,
    alpha: float = 0.05,
    rho: float = 0.0,
) -> GradeTest:
    """One-sided binomial test per grade: the probability of at least the observed defaults.

    ``rho = 0`` assumes independent defaults, which rejects too often when defaults move together.
    ``rho > 0`` uses the one-factor Vasicek model with that asset correlation (BCBS Working Paper 14, 2005,
    on the effect of default correlation on calibration tests).
    """
    n, d, p = _grade_inputs(n_obligors, n_defaults, pd)
    if not 0.0 <= rho < 1.0:
        raise ValueError("rho must lie in [0, 1)")
    name = "binomial" if rho == 0.0 else f"binomial (rho={rho})"
    return GradeTest(name, n, d, p, np.clip(_binomial_p(n, d, p, rho), 0.0, 1.0), alpha)


def jeffreys_test(
    n_obligors: ArrayLike, n_defaults: ArrayLike, pd: ArrayLike, alpha: float = 0.05
) -> GradeTest:
    """One-sided Jeffreys test per grade.

    With a Jeffreys prior the posterior of the default rate is Beta(d + 1/2, n - d + 1/2); the p-value is
    its CDF at the PD, i.e. the probability that the true default rate is below the PD. This is the PD
    back-test of the ECB instructions for reporting IRB validation results. For a portfolio-level test,
    pass the totals as one-element arrays with the obligor-weighted average PD.
    """
    n, d, p = _grade_inputs(n_obligors, n_defaults, pd)
    return GradeTest("jeffreys", n, d, p, np.asarray(stats.beta.cdf(p, d + 0.5, n - d + 0.5)), alpha)


Colour = Literal["green", "yellow", "red"]


@dataclass(frozen=True)
class TrafficLight:
    test: GradeTest
    colours: tuple[Colour, ...]
    thresholds: tuple[float, float]


def traffic_light(
    n_obligors: ArrayLike,
    n_defaults: ArrayLike,
    pd: ArrayLike,
    rho: float = 0.0,
    thresholds: tuple[float, float] = (0.95, 0.999),
) -> TrafficLight:
    """Green / yellow / red per grade from the (optionally correlated) binomial test.

    A grade is red when the observed defaults exceed the ``thresholds[1]`` quantile of the default count,
    yellow when they exceed the ``thresholds[0]`` quantile, green otherwise. The thresholds are a policy
    choice; set them in your validation policy. The approach follows Tasche (2003), "A traffic lights
    approach to PD validation".
    """
    lo, hi = (
        probability("thresholds[0]", thresholds[0]),
        probability("thresholds[1]", thresholds[1]),
    )
    if lo >= hi:
        raise ValueError("thresholds must be increasing")
    t = binomial_test(n_obligors, n_defaults, pd, rho=rho, alpha=1.0 - lo)
    colours: tuple[Colour, ...] = tuple(
        "red" if pv < 1.0 - hi else "yellow" if pv < 1.0 - lo else "green" for pv in t.p_value
    )
    return TrafficLight(t, colours, (lo, hi))


@dataclass(frozen=True)
class TestResult:
    test: str
    statistic: float
    p_value: float
    dof: int | None = None


def hosmer_lemeshow(n_obligors: ArrayLike, n_defaults: ArrayLike, pd: ArrayLike) -> TestResult:
    """Hosmer-Lemeshow test across all grades at once (two-sided), with ``K`` degrees of freedom for ``K``
    grades, the usual choice when the PDs were not fitted on the same sample."""
    n, d, p = _grade_inputs(n_obligors, n_defaults, pd)
    if np.any(n == 0):
        raise ValueError("every grade needs at least one obligor")
    expected = n * p
    statistic = float(np.sum((d - expected) ** 2 / (expected * (1.0 - p))))
    dof = int(n.size)
    return TestResult("hosmer_lemeshow", statistic, float(stats.chi2.sf(statistic, dof)), dof)


def normal_test(default_rates: ArrayLike, pd: ArrayLike) -> TestResult:
    """Normal test across years for one grade or portfolio (BCBS Working Paper 14, 2005).

    ``z = sum(DR_t - PD_t) / (sqrt(T) * tau)`` with ``tau`` the sample standard deviation of the yearly
    differences, and one-sided p-value ``1 - Phi(z)``. Unlike the per-year tests it does not assume
    independence across obligors; it assumes the yearly differences are independent across years.
    """
    dr = as_1d("default_rates", default_rates)
    p = as_1d("pd", pd)
    if dr.shape != p.shape or dr.size < 2:
        raise ValueError("need the same number (at least two) of yearly default rates and PDs")
    diff = dr - p
    tau = float(np.std(diff, ddof=1))
    if tau == 0.0:
        raise ValueError("the yearly differences have no spread")
    z = float(diff.sum() / (np.sqrt(diff.size) * tau))
    return TestResult("normal", z, float(stats.norm.sf(z)))


def spiegelhalter(defaults: ArrayLike, pd: ArrayLike) -> TestResult:
    """Spiegelhalter test on obligor-level PDs (two-sided).

    ``z = sum((y_i - p_i)(1 - 2 p_i)) / sqrt(sum((1 - 2 p_i)^2 p_i (1 - p_i)))``, standard normal when the
    PDs are right (Spiegelhalter, 1986). Unlike grade-level tests it uses every obligor's own PD.
    """
    y = as_1d("defaults", defaults)
    p = as_pd("pd", pd)
    if y.shape != p.shape:
        raise ValueError("defaults and pd must have the same length")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("defaults must contain only 0 and 1")
    num = float(np.sum((y - p) * (1.0 - 2.0 * p)))
    den = float(np.sqrt(np.sum((1.0 - 2.0 * p) ** 2 * p * (1.0 - p))))
    z = num / den
    return TestResult("spiegelhalter", z, float(2.0 * stats.norm.sf(abs(z))))
