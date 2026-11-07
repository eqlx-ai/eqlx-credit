import numpy as np
import pytest

from eqlx_credit import cohort_default_rates, long_run_average


def test_cohort_rates_by_hand():
    ref = ["2023-12-31"] * 4 + ["2024-12-31"] * 3
    dft = ["2024-06-30", None, "2025-03-01", "2024-12-31", "2025-12-31", None, "2026-01-01"]
    r = cohort_default_rates(ref, dft)
    assert [str(x) for x in r.reference_date] == ["2023-12-31", "2024-12-31"]
    # 2023 cohort: defaults on 2024-06-30 and 2024-12-31 (horizon end inclusive); 2025-03-01 is too late
    # 2024 cohort: 2025-12-31 counts (calendar year), 2026-01-01 does not
    assert list(r.n_obligors) == [4, 3] and list(r.n_defaults) == [2, 1]
    assert r.default_rate == pytest.approx([0.5, 1 / 3])
    assert long_run_average(r) == pytest.approx((0.5 + 1 / 3) / 2)
    assert long_run_average(r.default_rate, r.n_obligors, method="pooled") == pytest.approx(3 / 7)


def test_cohort_rates_by_grade():
    ref = ["2024-12-31"] * 5
    dft = [None, "2025-05-01", None, None, "2025-02-01"]
    grade = [1, 1, 2, 2, 2]
    r = cohort_default_rates(ref, dft, grade)
    assert [row["grade"] for row in r.rows()] == [1, 2]
    assert list(r.n_defaults) == [1, 1] and list(r.n_obligors) == [2, 3]
    with pytest.raises(ValueError):
        long_run_average(r)


def test_already_defaulted_obligor_is_an_input_error():
    with pytest.raises(ValueError):
        cohort_default_rates(["2024-12-31"], ["2024-11-30"])


def test_numpy_dates_and_errors():
    ref = np.array(["2024-01-01", "2024-01-01"], dtype="datetime64[D]")
    dft = np.array(["NaT", "2024-07-01"], dtype="datetime64[D]")
    assert list(cohort_default_rates(ref, dft).n_defaults) == [1]
    with pytest.raises(ValueError):
        long_run_average([0.1, 0.2], method="pooled")


def test_leap_day_reference_date():
    r = cohort_default_rates(["2024-02-29", "2024-02-29"], ["2025-02-28", "2025-03-01"])
    assert list(r.n_defaults) == [1]
