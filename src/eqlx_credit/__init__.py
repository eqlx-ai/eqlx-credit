"""eqlx-credit: open tools for credit-risk model work.

Validation, back-testing, calibration, master scales, low-default portfolio methods and the Basel IRB
risk-weight functions for probability-of-default (PD) models, built from public regulation and research.
"""

from .backtesting import (
    GradeTest,
    TestResult,
    TrafficLight,
    binomial_test,
    hosmer_lemeshow,
    jeffreys_test,
    normal_test,
    spiegelhalter,
    traffic_light,
)
from .calibration import (
    CurveFit,
    accuracy_ratio_from_grades,
    anchor_to_central_tendency,
    apply_pd_floor,
    fit_ct_ar,
    grade_positions,
)
from .cohorts import CohortRates, cohort_default_rates, long_run_average
from .discrimination import (
    AucResult,
    DeclineTest,
    auc,
    auc_ci,
    auc_ci_from_counts,
    auc_decline_test,
    auc_from_counts,
    gini,
    ks,
)
from .irb import asset_correlation, capital_requirement, risk_weight
from .ldp import VanDerBurgtFit, pluto_tasche, van_der_burgt
from .master_scale import MasterScale, geometric_master_scale, master_scale_from_midpoints
from .stability import hhi, psi

ChiSquareResult = TestResult  # v0.1 name

__all__ = [
    "AucResult",
    "ChiSquareResult",
    "CohortRates",
    "CurveFit",
    "DeclineTest",
    "GradeTest",
    "MasterScale",
    "TestResult",
    "TrafficLight",
    "VanDerBurgtFit",
    "accuracy_ratio_from_grades",
    "anchor_to_central_tendency",
    "apply_pd_floor",
    "asset_correlation",
    "auc",
    "auc_ci",
    "auc_ci_from_counts",
    "auc_decline_test",
    "auc_from_counts",
    "binomial_test",
    "capital_requirement",
    "cohort_default_rates",
    "fit_ct_ar",
    "geometric_master_scale",
    "gini",
    "grade_positions",
    "hhi",
    "hosmer_lemeshow",
    "jeffreys_test",
    "ks",
    "long_run_average",
    "master_scale_from_midpoints",
    "normal_test",
    "pluto_tasche",
    "psi",
    "risk_weight",
    "spiegelhalter",
    "traffic_light",
    "van_der_burgt",
]

__version__ = "0.2.0"
