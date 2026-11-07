"""Compatibility module: the v0.1 validation functions, now split by topic.

* discrimination: :mod:`eqlx_credit.discrimination`
* PD back-testing: :mod:`eqlx_credit.backtesting`
* stability and concentration: :mod:`eqlx_credit.stability`
"""

from .backtesting import GradeTest, TestResult, binomial_test, hosmer_lemeshow, jeffreys_test
from .discrimination import auc, gini, ks
from .stability import hhi, psi

ChiSquareResult = TestResult

__all__ = [
    "ChiSquareResult",
    "GradeTest",
    "TestResult",
    "auc",
    "binomial_test",
    "gini",
    "hhi",
    "hosmer_lemeshow",
    "jeffreys_test",
    "ks",
    "psi",
]
