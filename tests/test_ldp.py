import numpy as np
import pytest
from scipy import stats

from eqlx_credit import accuracy_ratio_from_grades, pluto_tasche, van_der_burgt
from eqlx_credit.ldp import _vasicek_cdf


def test_pluto_tasche_paper_example_independence():
    # Pluto & Tasche (2005), three grades with 100/400/300 obligors and no defaults,
    # 90% confidence: 0.29%, 0.33%, 0.76%.
    bounds = pluto_tasche([100, 400, 300], [0, 0, 0], confidence=0.9)
    assert bounds == pytest.approx(
        [1 - 0.1 ** (1 / 800), 1 - 0.1 ** (1 / 700), 1 - 0.1 ** (1 / 300)]
    )
    assert np.round(bounds * 100, 2) == pytest.approx([0.29, 0.33, 0.76])


def test_pluto_tasche_with_defaults_is_clopper_pearson_on_pooled_counts():
    bounds = pluto_tasche([100, 400, 300], [0, 2, 1], confidence=0.95)
    assert bounds[0] == pytest.approx(stats.beta.ppf(0.95, 4, 800 - 3))
    assert bounds[2] == pytest.approx(stats.beta.ppf(0.95, 2, 300 - 1))
    assert np.all(np.diff(bounds) >= 0)


def test_vasicek_cdf_reduces_to_binomial_without_correlation():
    assert _vasicek_cdf(3, 500, 0.01, 1e-9, 64) == pytest.approx(
        stats.binom.cdf(3, 500, 0.01), rel=1e-6
    )


def test_correlation_makes_bounds_more_prudent():
    n, d = [100, 400, 300], [0, 0, 0]
    b0 = pluto_tasche(n, d, 0.9, rho=0.0)
    b1 = pluto_tasche(n, d, 0.9, rho=0.12)
    b2 = pluto_tasche(n, d, 0.9, rho=0.24)
    assert np.all(b1 > b0) and np.all(b2 > b1)
    assert np.all(np.diff(b1) >= 0)
    # the bound is defined by: P(no more defaults than observed) = 1 - confidence
    assert _vasicek_cdf(0, 800, b1[0], 0.12, 96) == pytest.approx(0.1, abs=1e-9)


def test_pluto_tasche_all_defaulted_grade():
    assert pluto_tasche([10], [10])[0] == 1.0


def test_van_der_burgt_mean_and_shape():
    w = np.array([5, 10, 20, 25, 20, 12, 6, 2], dtype=float)
    fit = van_der_burgt(w, central_tendency=0.02, accuracy_ratio=0.6)
    assert np.dot(w / w.sum(), fit.pd) == pytest.approx(0.02, abs=1e-12)
    assert np.all(np.diff(fit.pd) > 0)
    # grading coarsens the continuous curve, so the grade-level AR is a bit lower
    ar = accuracy_ratio_from_grades(fit.pd, w)
    assert 0.45 < ar < 0.6


def test_van_der_burgt_continuous_ar():
    from eqlx_credit.ldp import _vdb_accuracy_ratio

    fit = van_der_burgt([1] * 50, 0.01, 0.7)
    assert _vdb_accuracy_ratio(fit.concavity, 0.01) == pytest.approx(0.7, abs=1e-9)
