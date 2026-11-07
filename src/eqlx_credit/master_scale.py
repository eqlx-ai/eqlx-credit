"""Master scale: the fixed table that turns a PD into a rating grade.

A bank's models produce a probability of default (PD) per obligor; the master scale
cuts the PD axis into grades so every model reports on the same ladder. The usual
design is geometric: each grade's PD is a constant multiple of the one above it, and
the boundary between two grades is the geometric mean of their PDs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._checks import as_pd, probability


@dataclass(frozen=True)
class MasterScale:
    """Grades are numbered from 1 (lowest risk) to ``len(midpoints)``.

    ``midpoints[i]`` is the representative PD of grade ``i + 1``; ``upper_bounds[i]``
    is the PD at which grade ``i + 1`` ends (the last bound is 1).
    """

    midpoints: NDArray[np.float64]
    upper_bounds: NDArray[np.float64]

    @property
    def lower_bounds(self) -> NDArray[np.float64]:
        return np.concatenate([[0.0], self.upper_bounds[:-1]])

    @property
    def n_grades(self) -> int:
        return int(self.midpoints.size)

    def grade(self, pd: ArrayLike) -> NDArray[np.int64]:
        """Grade number (1-based) for each PD. A PD equal to a boundary goes to the
        riskier grade."""
        p = as_pd("pd", pd, open_interval=False)
        idx = np.searchsorted(self.upper_bounds[:-1], p, side="right")
        return (idx + 1).astype(np.int64)

    def table(self) -> list[dict[str, float]]:
        """Rows of grade, lower bound, midpoint and upper bound, for printing."""
        return [
            {"grade": i + 1, "lower": float(lo), "midpoint": float(m), "upper": float(hi)}
            for i, (lo, m, hi) in enumerate(
                zip(self.lower_bounds, self.midpoints, self.upper_bounds, strict=True)
            )
        ]


def geometric_master_scale(pd_min: float, pd_max: float, n_grades: int) -> MasterScale:
    """Build a geometric master scale with ``n_grades`` performing grades.

    Example: ``geometric_master_scale(0.0003, 0.30, 20)``. The defaulted grade
    (PD = 1) is not part of the scale; keep it as a separate category.
    """
    lo = probability("pd_min", pd_min)
    hi = probability("pd_max", pd_max)
    if lo >= hi:
        raise ValueError("pd_min must be below pd_max")
    if n_grades < 2:
        raise ValueError("n_grades must be at least 2")
    midpoints = np.geomspace(lo, hi, n_grades)
    inner = np.sqrt(midpoints[:-1] * midpoints[1:])
    return MasterScale(midpoints=midpoints, upper_bounds=np.append(inner, 1.0))


def master_scale_from_midpoints(midpoints: ArrayLike) -> MasterScale:
    """Master scale from given grade PDs (strictly increasing), with geometric-mean
    boundaries between neighbours."""
    m = as_pd("midpoints", midpoints)
    if m.size < 2 or np.any(np.diff(m) <= 0):
        raise ValueError("midpoints must be at least two strictly increasing PDs")
    inner = np.sqrt(m[:-1] * m[1:])
    return MasterScale(midpoints=m, upper_bounds=np.append(inner, 1.0))
