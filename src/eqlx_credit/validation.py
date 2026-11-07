"""Validation statistics for probability-of-default (PD) models.

Three questions a validator asks of a rating model, and the tests here that answer them:

* Does it rank well? (discrimination)  -> :func:`auc`, :func:`gini`, :func:`ks`
* Are its PDs right? (calibration)      -> :func:`binomial_test`, :func:`jeffreys_test`,
  :func:`hosmer_lemeshow`
* Has the portfolio moved? (stability)  -> :func:`psi`, :func:`hhi`

Conventions: a *higher* score means *higher* risk; ``1`` in the default flag means default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

from ._checks import as_1d, as_counts, as_pd, obligors_and_defaults


def _split(defaults: ArrayLike, scores: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    y = as_1d("defaults", defaults)
    s = as_1d("scores", scores)
    if y.shape != s.shape:
        raise ValueError("defaults and scores must have the same length")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("defaults must contain only 0 and 1")
    bad, good = s[y == 1], s[y == 0]
    if bad.size == 0 or good.size == 0:
        raise ValueError("need at least one default and one non-default")
    return bad, good


def auc(defaults: ArrayLike, scores: ArrayLike) -> float:
    """Area under the ROC curve: the chance a random defaulter scores riskier than a
    random non-defaulter, counting ties as one half (Mann-Whitney U / n_D n_N)."""
    bad, good = _split(defaults, scores)
    ranks = stats.rankdata(np.concatenate([bad, good]))
    u = ranks[: bad.size].sum() - bad.size * (bad.size + 1) / 2.0
    return float(u / (bad.size * good.size))


def gini(defaults: ArrayLike, scores: ArrayLike) -> float:
    """Gini coefficient, equal to the accuracy ratio (AR): ``2 * AUC - 1``."""
    return 2.0 * auc(defaults, scores) - 1.0


def ks(defaults: ArrayLike, scores: ArrayLike) -> float:
    """Kolmogorov-Smirnov distance between the score distributions of defaulters and
    non-defaulters (largest gap between their cumulative distributions)."""
    bad, good = _split(defaults, scores)
    grid = np.unique(np.concatenate([bad, good]))
    cdf_bad = np.searchsorted(np.sort(bad), grid, side="right") / bad.size
    cdf_good = np.searchsorted(np.sort(good), grid, side="right") / good.size
    return float(np.max(np.abs(cdf_good - cdf_bad)))


@dataclass(frozen=True)
class GradeTest:
    """Per-grade result of a calibration test.

    ``p_value`` is small when the observed defaults are too many for the assigned PD,
    i.e. when the PD looks *underestimated*. ``rejected`` compares it with ``alpha``.
    """

    p_value: np.ndarray
    rejected: np.ndarray
    alpha: float


def binomial_test(
    n_obligors: ArrayLike, n_defaults: ArrayLike, pd: ArrayLike, alpha: float = 0.05
) -> GradeTest:
    """One-sided exact binomial test per grade: P(X >= d) with X ~ Binomial(n, PD).

    Assumes defaults are independent, so it rejects too often when defaults are
    correlated; read it together with :func:`jeffreys_test` and a correlation-aware view.
    """
    n, d = obligors_and_defaults(n_obligors, n_defaults)
    p = as_pd("pd", pd)
    if p.shape != n.shape:
        raise ValueError("pd must have one value per grade")
    p_value = stats.binom.sf(d - 1, n, p)
    return GradeTest(p_value=p_value, rejected=p_value < alpha, alpha=alpha)


def jeffreys_test(
    n_obligors: ArrayLike, n_defaults: ArrayLike, pd: ArrayLike, alpha: float = 0.05
) -> GradeTest:
    """One-sided Jeffreys test per grade.

    With a Jeffreys prior, the posterior of the default rate is
    Beta(d + 1/2, n - d + 1/2). The p-value is that posterior's CDF at the assigned PD:
    the probability that the true default rate is below the PD. This is the PD
    back-test the ECB asks for in its instructions for IRB validation reporting.
    """
    n, d = obligors_and_defaults(n_obligors, n_defaults)
    p = as_pd("pd", pd)
    if p.shape != n.shape:
        raise ValueError("pd must have one value per grade")
    p_value = stats.beta.cdf(p, d + 0.5, n - d + 0.5)
    return GradeTest(p_value=p_value, rejected=p_value < alpha, alpha=alpha)


@dataclass(frozen=True)
class ChiSquareResult:
    statistic: float
    p_value: float
    dof: int


def hosmer_lemeshow(n_obligors: ArrayLike, n_defaults: ArrayLike, pd: ArrayLike) -> ChiSquareResult:
    """Hosmer-Lemeshow test across all grades at once (two-sided).

    Uses ``K`` degrees of freedom for ``K`` grades, the usual choice when the PDs were
    not fitted on the same sample (out-of-sample back-testing).
    """
    n, d = obligors_and_defaults(n_obligors, n_defaults)
    p = as_pd("pd", pd)
    if p.shape != n.shape:
        raise ValueError("pd must have one value per grade")
    if np.any(n == 0):
        raise ValueError("every grade needs at least one obligor")
    expected = n * p
    statistic = float(np.sum((d - expected) ** 2 / (expected * (1.0 - p))))
    dof = int(n.size)
    return ChiSquareResult(statistic, float(stats.chi2.sf(statistic, dof)), dof)


def psi(expected: ArrayLike, actual: ArrayLike, epsilon: float = 1e-6) -> float:
    """Population stability index between two distributions over the same grades.

    Accepts counts or shares. Empty grades are floored at ``epsilon`` so the log is
    defined. A common reading: below 0.10 stable, 0.10-0.25 watch, above 0.25 shifted.
    """
    e = as_1d("expected", expected)
    a = as_1d("actual", actual)
    if e.shape != a.shape:
        raise ValueError("expected and actual must have the same length")
    if np.any(e < 0) or np.any(a < 0) or e.sum() == 0 or a.sum() == 0:
        raise ValueError("expected and actual must be non-negative and not all zero")
    e = np.maximum(e / e.sum(), epsilon)
    a = np.maximum(a / a.sum(), epsilon)
    return float(np.sum((a - e) * np.log(a / e)))


def hhi(n_obligors: ArrayLike, normalized: bool = True) -> float:
    """Herfindahl-Hirschman index of how obligors spread across grades.

    ``normalized=True`` rescales to 0 (perfectly even) .. 1 (everyone in one grade).
    """
    n = as_counts("n_obligors", n_obligors)
    if n.sum() == 0:
        raise ValueError("n_obligors must not be all zero")
    share = n / n.sum()
    value = float(np.sum(share**2))
    k = n.size
    if not normalized or k == 1:
        return value
    return (value - 1.0 / k) / (1.0 - 1.0 / k)
