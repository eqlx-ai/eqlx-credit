import numpy as np
import pytest

from eqlx_credit import asset_correlation, capital_requirement, risk_weight

# Basel II (June 2006), Annex 5 "Illustrative IRB Risk Weights for UL", M = 2.5 years.
# Columns: corporate LGD 45% (turnover 50m), SME LGD 45% (turnover 5m), residential mortgage LGD 45% and 25%,
# other retail LGD 45% and 85%, QRRE LGD 45% and 85%. Values in percent, as published (two decimals).
ANNEX5 = """
0.03 14.44 11.30 4.15 2.30 4.45 8.41 0.98 1.85
0.05 19.65 15.39 6.23 3.46 6.63 12.52 1.51 2.86
0.10 29.65 23.30 10.69 5.94 11.16 21.08 2.71 5.12
0.25 49.47 39.01 21.30 11.83 21.15 39.96 5.76 10.88
0.40 62.72 49.49 29.94 16.64 28.42 53.69 8.41 15.88
0.50 69.61 54.91 35.08 19.49 32.36 61.13 10.04 18.97
0.75 82.78 65.14 46.46 25.81 40.10 75.74 13.80 26.06
1.00 92.32 72.40 56.40 31.33 45.77 86.46 17.22 32.53
1.30 100.95 78.77 67.00 37.22 50.80 95.95 21.02 39.70
1.50 105.59 82.11 73.45 40.80 53.37 100.81 23.40 44.19
2.00 114.86 88.55 87.94 48.85 57.99 109.53 28.92 54.63
2.50 122.16 93.43 100.64 55.91 60.90 115.03 33.98 64.18
3.00 128.44 97.58 111.99 62.22 62.79 118.61 38.66 73.03
4.00 139.58 105.04 131.63 73.13 65.01 122.80 47.16 89.08
5.00 149.86 112.27 148.22 82.35 66.42 125.45 54.75 103.41
6.00 159.61 119.48 162.52 90.29 67.73 127.94 61.61 116.37
10.00 193.09 146.51 204.41 113.56 75.54 142.69 83.89 158.47
15.00 221.54 171.91 235.72 130.96 88.60 167.36 103.89 196.23
20.00 238.23 188.42 253.12 140.62 100.28 189.41 117.99 222.86
"""
TABLE = np.array([[float(x) for x in line.split()] for line in ANNEX5.strip().splitlines()])
PD = TABLE[:, 0] / 100
COLUMNS = [
    (1, "corporate", 0.45, None),
    (2, "corporate", 0.45, 5.0),
    (3, "residential_mortgage", 0.45, None),
    (4, "residential_mortgage", 0.25, None),
    (5, "other_retail", 0.45, None),
    (6, "other_retail", 0.85, None),
    (7, "qrre", 0.45, None),
    (8, "qrre", 0.85, None),
]


@pytest.mark.parametrize(("col", "asset_class", "lgd", "sales"), COLUMNS)
def test_reproduces_basel_annex_5(col, asset_class, lgd, sales):
    rw = risk_weight(PD, lgd, asset_class, sales_eur_m=sales) * 100
    # published to two decimals: allow one unit in the last place for rounding
    assert np.max(np.abs(rw - TABLE[:, col])) <= 0.0101


def test_maturity_adjustment():
    base = capital_requirement(0.01, 0.45)
    assert (
        capital_requirement(0.01, 0.45, maturity=1.0)[0]
        < base[0]
        < capital_requirement(0.01, 0.45, maturity=5.0)[0]
    )
    # maturity is clamped to 1..5 years and ignored outside corporates
    assert capital_requirement(0.01, 0.45, maturity=7.0) == pytest.approx(
        capital_requirement(0.01, 0.45, maturity=5.0)
    )
    mort = capital_requirement(0.01, 0.45, "residential_mortgage")
    assert capital_requirement(0.01, 0.45, "residential_mortgage", maturity=5.0) == pytest.approx(mort)


def test_correlations():
    assert asset_correlation(1e-9)[0] == pytest.approx(0.24, abs=1e-6)
    assert asset_correlation(0.999)[0] == pytest.approx(0.12, abs=1e-6)
    assert asset_correlation(0.01, sales_eur_m=5)[0] == pytest.approx(asset_correlation(0.01)[0] - 0.04)
    assert asset_correlation(0.01, sales_eur_m=1)[0] == pytest.approx(
        asset_correlation(0.01, sales_eur_m=5)[0]
    )
    assert asset_correlation(0.01, sales_eur_m=80)[0] == pytest.approx(asset_correlation(0.01)[0])
    assert asset_correlation([0.01, 0.2], "residential_mortgage") == pytest.approx([0.15, 0.15])
    with pytest.raises(ValueError):
        asset_correlation(0.01, "qrre", sales_eur_m=5)


def test_scaling_floor_and_errors():
    assert risk_weight(0.01, 0.45, scaling=1.06)[0] == pytest.approx(0.9232 * 1.06, abs=2e-4)
    assert risk_weight(0.0001, 0.45, pd_floor=0.0003)[0] == pytest.approx(risk_weight(0.0003, 0.45)[0])
    with pytest.raises(ValueError):
        risk_weight(1.0, 0.45)
    with pytest.raises(ValueError):
        risk_weight(0.01, 1.2)
