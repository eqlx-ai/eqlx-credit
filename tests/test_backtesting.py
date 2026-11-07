import math

import numpy as np
import pytest

from eqlx_credit import binomial_test, jeffreys_test, normal_test, spiegelhalter, traffic_light


def test_grade_test_report_rows():
    t = jeffreys_test([1000, 500], [12, 30], [0.01, 0.04])
    rows = t.rows()
    assert rows[1] == {
        "grade": 2,
        "obligors": 500,
        "defaults": 30,
        "pd": 0.04,
        "observed_rate": 0.06,
        "expected_defaults": 20.0,
        "p_value": pytest.approx(float(t.p_value[1])),
        "rejected": bool(t.rejected[1]),
    }


def test_correlated_binomial_reduces_to_independent():
    a = binomial_test([1000], [18], [0.01])
    b = binomial_test([1000], [18], [0.01], rho=1e-9)
    assert b.p_value[0] == pytest.approx(a.p_value[0], rel=1e-5)


def test_correlated_binomial_against_simulation():
    n, d, p, rho = 500, 12, 0.01, 0.12
    t = binomial_test([n], [d], [p], rho=rho)
    rng = np.random.default_rng(6)
    z = rng.standard_normal(200_000)
    from scipy import stats

    cond = stats.norm.cdf((stats.norm.ppf(p) - math.sqrt(rho) * z) / math.sqrt(1 - rho))
    sim = np.mean(rng.binomial(n, cond) >= d)
    assert t.p_value[0] == pytest.approx(sim, abs=0.003)
    # correlation makes the same excess of defaults much less surprising
    assert t.p_value[0] > binomial_test([n], [d], [p]).p_value[0] * 5


def test_zero_defaults_never_reject():
    assert binomial_test([100], [0], [0.05], rho=0.2).p_value[0] == pytest.approx(1.0)


def test_traffic_light():
    tl = traffic_light([1000, 1000, 1000], [10, 18, 30], [0.01, 0.01, 0.01])
    assert tl.colours == ("green", "yellow", "red")
    # with correlated defaults the same counts are far less surprising: all three stay green
    tl_rho = traffic_light([1000, 1000, 1000], [10, 18, 30], [0.01, 0.01, 0.01], rho=0.12)
    assert tl_rho.colours == ("green", "green", "green")
    assert traffic_light([1000, 1000], [60, 120], [0.01, 0.01], rho=0.12).colours == ("yellow", "red")
    with pytest.raises(ValueError):
        traffic_light([10], [1], [0.1], thresholds=(0.99, 0.95))


def test_normal_test_by_hand():
    dr = np.array([0.012, 0.018, 0.025, 0.015])
    pd = np.array([0.015, 0.015, 0.015, 0.015])
    diff = dr - pd
    z = diff.sum() / (2 * np.std(diff, ddof=1))
    r = normal_test(dr, pd)
    assert r.statistic == pytest.approx(z)
    assert r.p_value == pytest.approx(0.5 * math.erfc(z / math.sqrt(2)))
    with pytest.raises(ValueError):
        normal_test([0.01], [0.01])


def test_spiegelhalter_by_hand_and_under_correct_pds():
    y, p = np.array([0, 1, 0, 0]), np.array([0.1, 0.4, 0.2, 0.05])
    num = np.sum((y - p) * (1 - 2 * p))
    den = math.sqrt(np.sum((1 - 2 * p) ** 2 * p * (1 - p)))
    assert spiegelhalter(y, p).statistic == pytest.approx(num / den)
    rng = np.random.default_rng(7)
    zs = []
    for _ in range(300):
        pd = rng.uniform(0.001, 0.2, 2000)
        zs.append(spiegelhalter((rng.random(2000) < pd).astype(float), pd).statistic)
    assert abs(np.mean(zs)) < 0.15 and np.std(zs) == pytest.approx(1.0, abs=0.12)
