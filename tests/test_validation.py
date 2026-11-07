import math

import numpy as np
import pytest

from eqlx_credit import (
    auc,
    binomial_test,
    gini,
    hhi,
    hosmer_lemeshow,
    jeffreys_test,
    ks,
    psi,
)


def brute_force_auc(y, s):
    bad = [b for b, f in zip(s, y, strict=True) if f == 1]
    good = [g for g, f in zip(s, y, strict=True) if f == 0]
    wins = sum((b > g) + 0.5 * (b == g) for b in bad for g in good)
    return wins / (len(bad) * len(good))


def test_auc_matches_pairwise_count_with_ties():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 300)
    s = rng.integers(0, 8, 300) + y  # ties on purpose, defaulters a bit riskier
    assert auc(y, s) == pytest.approx(brute_force_auc(y, s), abs=1e-12)
    assert gini(y, s) == pytest.approx(2 * auc(y, s) - 1)


def test_auc_extremes():
    assert auc([0, 0, 1, 1], [1, 2, 3, 4]) == 1.0
    assert auc([1, 1, 0, 0], [1, 2, 3, 4]) == 0.0
    assert auc([0, 1], [5, 5]) == 0.5


def test_ks_known_value():
    # defaulters at 3,4 ; non-defaulters at 1,2,3 -> largest CDF gap at score 2: 2/3 - 0
    assert ks([0, 0, 0, 1, 1], [1, 2, 3, 3, 4]) == pytest.approx(2 / 3)


def test_input_errors():
    with pytest.raises(ValueError):
        auc([0, 0], [1, 2])
    with pytest.raises(ValueError):
        auc([0, 2], [1, 2])
    with pytest.raises(ValueError):
        binomial_test([10], [11], [0.1])


def test_binomial_test_exact_value():
    n, d, p = 10, 3, 0.1
    expected = sum(math.comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(d, n + 1))
    res = binomial_test([n], [d], [p])
    assert res.p_value[0] == pytest.approx(expected, rel=1e-12)
    assert binomial_test([10], [0], [0.1]).p_value[0] == pytest.approx(1.0)


def test_jeffreys_test_direction_and_rejection():
    low = jeffreys_test([1000], [5], [0.02])  # far fewer defaults than PD implies
    high = jeffreys_test([1000], [40], [0.02])  # far more defaults than PD implies
    assert low.p_value[0] > 0.99 and not low.rejected[0]
    assert high.p_value[0] < 0.001 and high.rejected[0]
    mid = jeffreys_test([100000], [2000], [0.02])
    assert mid.p_value[0] == pytest.approx(0.5, abs=0.02)


def test_hosmer_lemeshow():
    perfect = hosmer_lemeshow([100, 200], [1, 10], [0.01, 0.05])
    assert perfect.statistic == pytest.approx(0.0)
    assert perfect.p_value == pytest.approx(1.0)
    assert perfect.dof == 2
    off = hosmer_lemeshow([1000, 1000], [30, 100], [0.01, 0.05])
    assert off.p_value < 1e-6


def test_psi():
    assert psi([10, 20, 30], [1, 2, 3]) == pytest.approx(0.0)
    e, a = np.array([0.5, 0.5]), np.array([0.25, 0.75])
    expected = float(np.sum((a - e) * np.log(a / e)))
    assert psi(e, a) == pytest.approx(expected)
    with pytest.warns(UserWarning, match="floored"):
        assert psi([1, 1], [1, 0]) > 1.0  # an emptied grade is a large shift, not an error


def test_hhi():
    assert hhi([10, 10, 10, 10]) == pytest.approx(0.0)
    assert hhi([0, 40, 0, 0]) == pytest.approx(1.0)
    assert hhi([1, 1], normalized=False) == pytest.approx(0.5)


def test_exposure_weighted_hhi():
    assert hhi([10, 10], exposure=[100, 0]) == pytest.approx(1.0)
    assert hhi([1, 99], exposure=[50, 50]) == pytest.approx(0.0)
