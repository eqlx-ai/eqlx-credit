import numpy as np
import pytest
from scipy.special import logit

from eqlx_credit import (
    accuracy_ratio_from_grades,
    anchor_to_central_tendency,
    auc,
    fit_ct_ar,
)


def test_anchor_hits_target_and_keeps_ranking_and_spread():
    rng = np.random.default_rng(7)
    pd = 1 / (1 + np.exp(-rng.normal(-4, 1, 5000)))
    out = anchor_to_central_tendency(pd, 0.03)
    assert out.mean() == pytest.approx(0.03, abs=1e-12)
    assert np.array_equal(np.argsort(pd), np.argsort(out))
    shift = logit(out) - logit(pd)
    assert np.allclose(shift, shift[0])  # same log-odds shift for everyone


def test_anchor_with_grade_weights():
    out = anchor_to_central_tendency([0.01, 0.05, 0.2], 0.04, weights=[600, 300, 100])
    assert np.dot([0.6, 0.3, 0.1], out) == pytest.approx(0.04, abs=1e-12)


def test_grade_ar_equals_obligor_level_gini():
    n = np.array([400, 300, 200, 100])
    d = np.array([2, 6, 12, 20])
    y = np.concatenate([np.r_[np.ones(k), np.zeros(m - k)] for m, k in zip(n, d, strict=True)])
    s = np.repeat(np.arange(4), n)
    assert accuracy_ratio_from_grades(d / n, n) == pytest.approx(2 * auc(y, s) - 1, abs=1e-12)


def test_fit_ct_ar_recovers_targets():
    w = [5, 10, 20, 25, 20, 12, 6, 2]
    fit = fit_ct_ar(w, central_tendency=0.02, accuracy_ratio=0.6)
    assert fit.central_tendency == pytest.approx(0.02, abs=1e-10)
    assert fit.accuracy_ratio == pytest.approx(0.6, abs=1e-8)
    assert np.all(np.diff(fit.pd) > 0)
    assert fit.slope > 0


def test_fit_ct_ar_unreachable():
    with pytest.raises(ValueError):
        fit_ct_ar([50, 50], central_tendency=0.02, accuracy_ratio=0.99)
