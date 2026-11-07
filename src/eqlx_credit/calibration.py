"""Calibration: moving a model's PDs to the level the portfolio should have.

A score model ranks obligors; calibration decides how high their PDs are. Two tools:

* :func:`anchor_to_central_tendency` keeps the ranking and the spread of the model and
  shifts every PD on the log-odds scale until the portfolio average equals a target
  (the *central tendency*, e.g. a long-run average default rate).
* :func:`fit_ct_ar` builds a whole PD curve over rating grades from two targets only:
  the central tendency and the accuracy ratio (AR) the model should have. It follows
  the idea of Tasche's quasi-moment-matching (QMM) approach, a two-parameter logistic
  PD curve, but matches the AR exactly on the discrete grade distribution instead of
  through a moment approximation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import brentq
from scipy.special import expit, logit

from ._checks import as_1d, as_pd, probability


def _weights(weights: ArrayLike | None, size: int) -> NDArray[np.float64]:
    if weights is None:
        return np.full(size, 1.0 / size)
    w = as_1d("weights", weights)
    if w.size != size:
        raise ValueError("weights must have one value per PD")
    if np.any(w < 0) or w.sum() == 0:
        raise ValueError("weights must be non-negative and not all zero")
    return w / w.sum()


def anchor_to_central_tendency(
    pd: ArrayLike, target: float, weights: ArrayLike | None = None
) -> NDArray[np.float64]:
    """Shift PDs on the log-odds scale so their weighted average equals ``target``.

    ``pd`` can be one value per obligor (no weights) or one per grade with
    ``weights`` set to the grade's obligor count. The ranking is unchanged.
    """
    p = as_pd("pd", pd)
    w = _weights(weights, p.size)
    t = probability("target", target)
    z = logit(p)

    def gap(shift: float) -> float:
        return float(np.dot(w, expit(z + shift)) - t)

    shift = brentq(gap, -60.0, 60.0, xtol=1e-14)
    return np.asarray(expit(z + shift), dtype=float)


def accuracy_ratio_from_grades(pd: ArrayLike, weights: ArrayLike) -> float:
    """Accuracy ratio implied by grade PDs and the share of obligors in each grade.

    Grades are ordered from lowest to highest risk. Obligors in the same grade count as
    ties (half a correct ranking), as in a CAP or ROC curve drawn on grades.
    """
    p = as_pd("pd", pd, open_interval=False)
    w = _weights(weights, p.size)
    mean = float(np.dot(w, p))
    if not 0.0 < mean < 1.0:
        raise ValueError("the portfolio must contain both defaults and non-defaults")
    f_bad = w * p / mean
    f_good = w * (1.0 - p) / (1.0 - mean)
    good_below = np.concatenate([[0.0], np.cumsum(f_good)[:-1]])
    area = float(np.sum(f_bad * (good_below + 0.5 * f_good)))
    return 2.0 * area - 1.0


@dataclass(frozen=True)
class CurveFit:
    """Logistic PD curve ``PD_i = expit(intercept + slope * x_i)`` over grades, where
    ``x_i`` runs evenly from 0 (lowest-risk grade) to 1 (highest-risk grade)."""

    pd: NDArray[np.float64]
    intercept: float
    slope: float
    central_tendency: float
    accuracy_ratio: float


def fit_ct_ar(weights: ArrayLike, central_tendency: float, accuracy_ratio: float) -> CurveFit:
    """Fit grade PDs to a target central tendency and accuracy ratio.

    ``weights`` is the number (or share) of obligors per grade, lowest risk first.
    Useful when defaults are too few to estimate grade PDs directly: the shape comes
    from the target AR, the level from the central tendency.
    """
    w = _weights(weights, np.asarray(weights).size)
    if w.size < 2:
        raise ValueError("need at least two grades")
    ct = probability("central_tendency", central_tendency)
    target_ar = probability("accuracy_ratio", accuracy_ratio)
    x = np.linspace(0.0, 1.0, w.size)

    def intercept_for(slope: float) -> float:
        return float(
            brentq(lambda a: float(np.dot(w, expit(a + slope * x)) - ct), -80.0, 80.0, xtol=1e-14)
        )

    def ar_for(slope: float) -> float:
        return accuracy_ratio_from_grades(expit(intercept_for(slope) + slope * x), w)

    slope_hi = 1.0
    while ar_for(slope_hi) < target_ar:
        slope_hi *= 2.0
        if slope_hi > 200.0:
            raise ValueError(
                f"accuracy ratio {target_ar} is not reachable with this grade distribution"
            )
    slope = brentq(lambda b: ar_for(b) - target_ar, 0.0, slope_hi, xtol=1e-12)
    intercept = intercept_for(slope)
    pd = np.asarray(expit(intercept + slope * x), dtype=float)
    return CurveFit(
        pd=pd,
        intercept=float(intercept),
        slope=float(slope),
        central_tendency=float(np.dot(w, pd)),
        accuracy_ratio=accuracy_ratio_from_grades(pd, w),
    )
