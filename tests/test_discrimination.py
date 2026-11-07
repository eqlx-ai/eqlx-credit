import numpy as np
import pytest

from eqlx_credit import auc, auc_ci, auc_ci_from_counts, auc_decline_test, auc_from_counts, gini, ks


def delong_brute(y, s):
    """DeLong variance by explicit placement values, obligor by obligor."""
    bad, good = s[y == 1], s[y == 0]
    v10 = np.array([np.mean((b > good) + 0.5 * (b == good)) for b in bad])
    v01 = np.array([np.mean((bad > g) + 0.5 * (bad == g)) for g in good])
    return v10.mean(), v10.var(ddof=1) / bad.size + v01.var(ddof=1) / good.size


def test_delong_matches_brute_force_with_ties():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 400)
    s = rng.integers(0, 10, 400) + 2 * y
    r = auc_ci(y, s)
    a, v = delong_brute(y, s)
    assert r.auc == pytest.approx(a, abs=1e-12)
    assert r.std_error == pytest.approx(np.sqrt(v), rel=1e-10)
    assert r.gini == pytest.approx(2 * a - 1)


def test_grade_table_equals_obligor_level():
    n = np.array([400, 300, 200, 100])
    d = np.array([2, 6, 12, 20])
    y = np.concatenate([np.r_[np.ones(k), np.zeros(m - k)] for m, k in zip(n, d, strict=True)])
    s = np.repeat(np.arange(4), n)
    assert auc_from_counts(n, d) == pytest.approx(auc(y, s), abs=1e-12)
    a, b = auc_ci_from_counts(n, d), auc_ci(y, s)
    assert (a.auc, a.std_error) == pytest.approx((b.auc, b.std_error))


def test_score_direction():
    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, 300)
    s = rng.normal(size=300) + y
    assert auc(y, -s, higher_is_riskier=False) == pytest.approx(auc(y, s))
    assert gini(y, s, higher_is_riskier=False) == pytest.approx(-gini(y, s))
    assert ks(y, s, higher_is_riskier=False) == pytest.approx(ks(y, s))


def test_standard_error_and_coverage_by_simulation():
    rng = np.random.default_rng(5)
    true_auc = 0.7602499389065233  # binormal, separation 1: Phi(1 / sqrt(2))
    aucs, ses, covered = [], [], 0
    for _ in range(400):
        y = (rng.random(600) < 0.1).astype(float)
        s = rng.normal(size=600) + y
        r = auc_ci(y, s)
        aucs.append(r.auc)
        ses.append(r.std_error)
        covered += r.ci_low <= true_auc <= r.ci_high
    assert np.mean(ses) == pytest.approx(np.std(aucs), rel=0.12)
    assert 0.91 <= covered / 400 <= 0.98


def test_decline_test():
    cur = auc_ci_from_counts([400, 300, 200, 100], [2, 6, 12, 20])
    same = auc_decline_test(cur.auc, cur)
    assert same.statistic == pytest.approx(0.0) and same.p_value == pytest.approx(0.5)
    worse = auc_decline_test(cur.auc + 3 * cur.std_error, cur)
    assert worse.rejected and worse.p_value == pytest.approx(0.00135, abs=1e-4)
    better = auc_decline_test(cur.auc - 0.05, cur)
    assert not better.rejected and better.p_value > 0.9
    with pytest.raises(ValueError):
        auc_decline_test(1.5, cur)
