# Changelog

## 0.2.0

Added
- `auc_ci`, `auc_ci_from_counts`: DeLong standard error and confidence interval; `auc_from_counts`.
- `auc_decline_test`: the ECB test of current against initial AUC.
- `binomial_test(..., rho=)`: correlated defaults (one-factor Vasicek); `traffic_light`; `normal_test`;
  `spiegelhalter`. Per-grade results carry obligors, defaults, observed rate and expected defaults, and
  `rows()` for report tables.
- `cohort_default_rates` (calendar-year horizon) and `long_run_average`.
- `risk_weight`, `capital_requirement`, `asset_correlation`: Basel IRB, reproducing Basel II Annex 5.
- `apply_pd_floor`, `grade_positions`; `hhi(..., exposure=)`.

Changed
- `auc`, `gini`, `ks` take `higher_is_riskier` (default `True`).
- `fit_ct_ar` places grades at Tasche's robust-logit positions by default (`placement="even"` for 0.1).
- `MasterScale.grade` maps PD = 1 to a separate defaulted grade (`default_grade`).
- `psi` warns when an empty grade is floored.
- Functions are grouped in `discrimination`, `backtesting`, `stability`, `cohorts`, `irb`;
  `eqlx_credit.validation` still imports the 0.1 names.

## 0.1.0

First release: AUC, Gini, KS; binomial, Jeffreys, Hosmer–Lemeshow; PSI, HHI; anchoring to a central
tendency; CT/AR curve; geometric master scale; Pluto–Tasche; van der Burgt.
