import numpy as np
import pytest

from eqlx_credit import geometric_master_scale, master_scale_from_midpoints


def test_geometric_grid():
    scale = geometric_master_scale(0.0003, 0.30, 20)
    assert scale.n_grades == 20
    assert scale.midpoints[0] == pytest.approx(0.0003)
    assert scale.midpoints[-1] == pytest.approx(0.30)
    ratios = scale.midpoints[1:] / scale.midpoints[:-1]
    assert np.allclose(ratios, ratios[0])
    assert scale.upper_bounds[-1] == 1.0
    assert np.all(scale.lower_bounds < scale.midpoints)
    assert np.all(scale.midpoints < scale.upper_bounds)


def test_grade_assignment():
    scale = master_scale_from_midpoints([0.01, 0.04, 0.16])
    # boundaries: sqrt(0.01*0.04)=0.02, sqrt(0.04*0.16)=0.08
    assert list(scale.grade([0.0, 0.01, 0.0199, 0.02, 0.05, 0.08, 0.5, 0.999])) == [1, 1, 1, 2, 2, 3, 3, 3]
    # a defaulted obligor never lands in a performing grade
    assert scale.default_grade == 4
    assert list(scale.grade([1.0, 0.5])) == [4, 3]
    assert list(scale.grade(scale.midpoints)) == [1, 2, 3]
    assert scale.table()[1] == pytest.approx({"grade": 2, "lower": 0.02, "midpoint": 0.04, "upper": 0.08})


def test_bad_inputs():
    with pytest.raises(ValueError):
        geometric_master_scale(0.1, 0.01, 5)
    with pytest.raises(ValueError):
        master_scale_from_midpoints([0.02, 0.01])
