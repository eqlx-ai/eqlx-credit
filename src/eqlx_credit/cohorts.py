"""One-year default rates by cohort, and their long-run average.

The cohort method: at each reference date, take the obligors that are performing (not in default) and
count how many default within the following year. The long-run average default rate is the arithmetic
average of those one-year rates over the observation period (EBA Guidelines on PD estimation, LGD
estimation and the treatment of defaulted exposures, EBA/GL/2017/16, section on observed average default
rates and the long-run average).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class CohortRates:
    """One row per reference date (and grade, if given)."""

    reference_date: NDArray[np.datetime64]
    grade: NDArray[Any] | None
    n_obligors: NDArray[np.int64]
    n_defaults: NDArray[np.int64]

    @property
    def default_rate(self) -> NDArray[np.float64]:
        return self.n_defaults / self.n_obligors

    def rows(self) -> list[dict[str, Any]]:
        return [
            {
                "reference_date": str(self.reference_date[i]),
                **({"grade": self.grade[i].item()} if self.grade is not None else {}),
                "obligors": int(self.n_obligors[i]),
                "defaults": int(self.n_defaults[i]),
                "default_rate": float(self.default_rate[i]),
            }
            for i in range(self.n_obligors.size)
        ]


def _dates(name: str, values: ArrayLike) -> NDArray[np.datetime64]:
    arr = np.asarray(values)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    try:
        return arr.astype("datetime64[D]")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must hold dates (ISO strings, date objects or datetime64)") from exc


def _add_years(dates: NDArray[np.datetime64], years: int) -> NDArray[np.datetime64]:
    """Same calendar day ``years`` later; 29 February maps to 28 February."""
    out = []
    for d in dates.astype(object):
        try:
            out.append(d.replace(year=d.year + years))
        except ValueError:  # 29 February into a non-leap year
            out.append(d.replace(year=d.year + years, day=28))
    return np.array(out, dtype="datetime64[D]")


def cohort_default_rates(
    reference_date: ArrayLike,
    default_date: ArrayLike,
    grade: ArrayLike | None = None,
    horizon_years: int = 1,
) -> CohortRates:
    """One-year (or ``horizon_years``) default rates per reference date, optionally per grade.

    Each row is one obligor performing at one reference date: ``reference_date``, the date of its next
    default after that date (``None``/``NaT`` if none), and optionally its grade at the reference date. An
    obligor defaults in the cohort if ``reference_date < default_date <= reference_date + horizon``, the
    horizon ending on the same calendar day ``horizon_years`` later (29 February ends on 28 February).
    Rows where the obligor was already in default at the reference date are an input error.
    """
    ref = _dates("reference_date", reference_date)
    dft = _dates("default_date", default_date)
    if dft.shape != ref.shape:
        raise ValueError("reference_date and default_date must have the same length")
    if np.any(~np.isnat(dft) & (dft <= ref)):
        raise ValueError("an obligor already in default at its reference date is not in the cohort")
    if horizon_years < 1:
        raise ValueError("horizon_years must be at least 1")
    end = _add_years(ref, horizon_years)
    hit = ~np.isnat(dft) & (dft <= end)

    if grade is None:
        keys, inv = np.unique(ref, return_inverse=True)
        n = np.bincount(inv)
        d = np.bincount(inv, weights=hit).astype(np.int64)
        return CohortRates(keys, None, n.astype(np.int64), d)
    g = np.asarray(grade)
    if g.shape != ref.shape:
        raise ValueError("grade must have one value per row")
    pairs = np.rec.fromarrays([ref, g], names="ref,grade")
    keys, inv = np.unique(pairs, return_inverse=True)
    n = np.bincount(inv)
    d = np.bincount(inv, weights=hit).astype(np.int64)
    return CohortRates(keys["ref"], keys["grade"], n.astype(np.int64), d)


def long_run_average(
    default_rates: ArrayLike | CohortRates,
    n_obligors: ArrayLike | None = None,
    method: Literal["average", "pooled"] = "average",
) -> float:
    """Long-run average default rate.

    ``"average"`` (the EBA guideline's default): the arithmetic mean of the yearly rates, each year weighing
    the same. ``"pooled"``: total defaults over total obligors, so large years weigh more (needs
    ``n_obligors``). Pass a :class:`CohortRates` without grades to use its rows directly.
    """
    if isinstance(default_rates, CohortRates):
        if default_rates.grade is not None:
            raise ValueError("compute the long-run average per grade, or use cohorts without grades")
        rates = default_rates.default_rate
        n_obligors = default_rates.n_obligors
    else:
        rates = np.asarray(default_rates, dtype=float)
    if rates.ndim != 1 or rates.size == 0 or np.any((rates < 0) | (rates > 1)):
        raise ValueError("default_rates must be a non-empty list of rates between 0 and 1")
    if method == "average":
        return float(rates.mean())
    if n_obligors is None:
        raise ValueError("method='pooled' needs n_obligors")
    n = np.asarray(n_obligors, dtype=float)
    if n.shape != rates.shape:
        raise ValueError("n_obligors must have one value per rate")
    return float(np.dot(rates, n) / n.sum())
