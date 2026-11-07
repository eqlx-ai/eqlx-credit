"""Discriminatory power: does the model rank defaulters above non-defaulters?

* :func:`auc`, :func:`gini`, :func:`ks` on obligor-level data, :func:`auc_from_counts` on a grade table.
* :func:`auc_ci` and :func:`auc_ci_from_counts` add the DeLong standard error and a confidence interval.
* :func:`auc_decline_test` is the test in the ECB's instructions for reporting IRB validation results: has
  the AUC fallen significantly below its value at initial validation?

By default a higher score means higher risk; pass ``higher_is_riskier=False`` for scorecards where a high
score means a safe obligor. Grades in a grade table run from lowest to highest risk.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats

from ._checks import as_1d, obligors_and_defaults, probability


def _levels(
    defaults: ArrayLike, scores: ArrayLike, higher_is_riskier: bool
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Defaults and non-defaults per distinct score, ordered from lowest to highest risk."""
    y = as_1d("defaults", defaults)
    s = as_1d("scores", scores)
    if y.shape != s.shape:
        raise ValueError("defaults and scores must have the same length")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("defaults must contain only 0 and 1")
    if not higher_is_riskier:
        s = -s
    uniq, idx = np.unique(s, return_inverse=True)
    d = np.bincount(idx, weights=y, minlength=uniq.size)
    g = np.bincount(idx, weights=1.0 - y, minlength=uniq.size)
    return d, g


def _delong(d: NDArray[np.float64], g: NDArray[np.float64]) -> tuple[float, float]:
    """AUC and its DeLong variance from defaults ``d`` and non-defaults ``g`` per level (ascending risk).

    Each defaulter's placement value is the share of non-defaulters it outranks (ties count one half); each
    non-defaulter's is the share of defaulters that outrank it. The AUC is their mean and the variance is
    ``var(V10) / m + var(V01) / n`` (DeLong, DeLong and Clarke-Pearson, 1988).
    """
    m, n = d.sum(), g.sum()
    if m == 0 or n == 0:
        raise ValueError("need at least one default and one non-default")
    g_below = np.cumsum(g) - g
    d_above = m - np.cumsum(d)
    v10 = (g_below + 0.5 * g) / n
    v01 = (d_above + 0.5 * d) / m
    auc = float(np.dot(d, v10) / m)
    s10 = float(np.dot(d, (v10 - auc) ** 2) / (m - 1)) if m > 1 else 0.0
    s01 = float(np.dot(g, (v01 - auc) ** 2) / (n - 1)) if n > 1 else 0.0
    return auc, s10 / m + s01 / n


def auc(defaults: ArrayLike, scores: ArrayLike, higher_is_riskier: bool = True) -> float:
    """Area under the ROC curve: the chance a random defaulter is ranked riskier than a random
    non-defaulter, ties counting one half."""
    return _delong(*_levels(defaults, scores, higher_is_riskier))[0]


def gini(defaults: ArrayLike, scores: ArrayLike, higher_is_riskier: bool = True) -> float:
    """Gini coefficient, equal to the accuracy ratio (AR): ``2 * AUC - 1``."""
    return 2.0 * auc(defaults, scores, higher_is_riskier) - 1.0


def ks(defaults: ArrayLike, scores: ArrayLike, higher_is_riskier: bool = True) -> float:
    """Kolmogorov-Smirnov distance: the largest gap between the cumulative score distributions of
    defaulters and non-defaulters. Direction does not change it."""
    d, g = _levels(defaults, scores, higher_is_riskier)
    return float(np.max(np.abs(np.cumsum(g) / g.sum() - np.cumsum(d) / d.sum())))


def auc_from_counts(n_obligors: ArrayLike, n_defaults: ArrayLike) -> float:
    """AUC from a grade table (obligors and defaults per grade, lowest risk first)."""
    n, d = obligors_and_defaults(n_obligors, n_defaults)
    return _delong(d, n - d)[0]


@dataclass(frozen=True)
class AucResult:
    """AUC with its DeLong standard error and a normal-approximation confidence interval."""

    auc: float
    std_error: float
    ci_low: float
    ci_high: float
    confidence: float
    n_defaults: int
    n_non_defaults: int

    @property
    def gini(self) -> float:
        return 2.0 * self.auc - 1.0

    @property
    def gini_ci(self) -> tuple[float, float]:
        return 2.0 * self.ci_low - 1.0, 2.0 * self.ci_high - 1.0


def _result(d: NDArray[np.float64], g: NDArray[np.float64], confidence: float) -> AucResult:
    c = probability("confidence", confidence)
    value, var = _delong(d, g)
    se = float(np.sqrt(var))
    z = float(stats.norm.ppf(0.5 + c / 2.0))
    return AucResult(
        auc=value,
        std_error=se,
        ci_low=max(0.0, value - z * se),
        ci_high=min(1.0, value + z * se),
        confidence=c,
        n_defaults=int(d.sum()),
        n_non_defaults=int(g.sum()),
    )


def auc_ci(
    defaults: ArrayLike, scores: ArrayLike, confidence: float = 0.95, higher_is_riskier: bool = True
) -> AucResult:
    """AUC on obligor-level data with its DeLong standard error and confidence interval."""
    return _result(*_levels(defaults, scores, higher_is_riskier), confidence)


def auc_ci_from_counts(n_obligors: ArrayLike, n_defaults: ArrayLike, confidence: float = 0.95) -> AucResult:
    """AUC on a grade table with its DeLong standard error; obligors in the same grade are ties."""
    n, d = obligors_and_defaults(n_obligors, n_defaults)
    return _result(d, n - d, confidence)


@dataclass(frozen=True)
class DeclineTest:
    auc_initial: float
    auc_current: float
    statistic: float
    p_value: float
    rejected: bool
    alpha: float


def auc_decline_test(auc_initial: float, current: AucResult, alpha: float = 0.05) -> DeclineTest:
    """Has discriminatory power fallen since initial validation?

    ``S = (AUC_initial - AUC_current) / s_current`` with ``s_current`` the DeLong standard error of the
    current AUC, and one-sided p-value ``1 - Phi(S)``: a small p-value means the AUC is significantly lower
    than at initial validation. This is the test of the ECB instructions for reporting the validation
    results of IRB models (2019), section on the current versus initial AUC.
    """
    if not 0.0 <= auc_initial <= 1.0:
        raise ValueError("auc_initial must lie in [0, 1]")
    if current.std_error <= 0:
        raise ValueError("the current AUC has no positive standard error")
    s = (auc_initial - current.auc) / current.std_error
    p = float(stats.norm.sf(s))
    return DeclineTest(auc_initial, current.auc, float(s), p, p < alpha, alpha)
