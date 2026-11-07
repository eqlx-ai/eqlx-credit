# eqlx-credit

Open tools for credit-risk model work: discriminatory power with confidence intervals, PD back-testing
that allows for correlated defaults, cohort default rates, calibration, master scales, low-default
portfolio methods and the Basel IRB risk-weight functions.

Built by [eqlx](https://eqlx.ai) from public regulation and published research. Part of
[eqlx.org](https://eqlx.org).

> Status: alpha (0.2). Every function is tested against a closed form, a published table or worked
> example, a brute-force calculation, or a simulation. The API may still change.

## Install

```bash
pip install git+https://github.com/eqlx-ai/eqlx-credit
```

Python 3.10+, depends only on NumPy and SciPy.

## What is inside

| Area | Function | What it answers | Source |
|---|---|---|---|
| Discrimination | `auc`, `gini`, `ks`, `auc_from_counts` | Does the model rank defaulters above non-defaulters? | |
| | `auc_ci`, `auc_ci_from_counts` | How precise is that AUC? (DeLong standard error) | DeLong et al. (1988) |
| | `auc_decline_test` | Has the AUC fallen significantly since initial validation? | ECB validation reporting instructions (2019) |
| PD back-testing | `jeffreys_test` | Is each grade's PD consistent with its defaults? | ECB validation reporting instructions (2019) |
| | `binomial_test(..., rho=)` | The same, optionally allowing for correlated defaults | BCBS WP14 (2005), Vasicek |
| | `traffic_light` | Green / yellow / red per grade at thresholds you set | Tasche (2003) |
| | `hosmer_lemeshow` | Are the PDs right across all grades at once? | |
| | `normal_test` | Are the PDs right across several years? | BCBS WP14 (2005) |
| | `spiegelhalter` | Are obligor-level PDs right? | Spiegelhalter (1986) |
| Default rates | `cohort_default_rates`, `long_run_average` | One-year default rates per cohort and their long-run average | EBA/GL/2017/16 |
| Stability | `psi`, `hhi(..., exposure=)` | Has the portfolio moved? Is it concentrated (by count or exposure)? | Siddiqi (2006) |
| Calibration | `anchor_to_central_tendency` | Shift PDs to a target average, keeping the ranking | |
| | `fit_ct_ar`, `grade_positions` | Grade PDs from a central tendency and an accuracy ratio | Tasche (2013) |
| | `apply_pd_floor` | Regulatory PD floor | Basel II, CRR, Basel III |
| Low-default portfolios | `pluto_tasche` | Most prudent PD upper bounds, independent or correlated defaults | Pluto & Tasche (2005) |
| | `van_der_burgt` | Grade PDs from a fitted CAP curve | van der Burgt (2008) |
| Master scale | `geometric_master_scale`, `master_scale_from_midpoints` | The PD-to-grade ladder, with a separate defaulted grade | |
| Capital | `risk_weight`, `capital_requirement`, `asset_correlation` | Basel IRB risk weights: corporate (with SME adjustment and maturity), mortgage, QRRE, other retail | Basel II ¶272–330, Annex 5 |

Conventions: grades run from lowest to highest risk; by default a higher score means higher risk
(`higher_is_riskier=False` for scorecards where a high score is safe); `1` in a default flag means
default; a PD of 1 maps to the master scale's defaulted grade, never to a performing grade.

## Agent skill: annual PD validation

[`skills/annual-pd-validation`](skills/annual-pd-validation) is an Agent Skill that lets an AI agent prepare the
evidence for an annual PD validation: it runs every test through eqlx-credit, reads the results, and drafts the
report in the user's language. The agent never computes a number itself and never gives the validation opinion; the
conclusion and signature stay with the validator. On the bundled synthetic portfolio it finds the planted problems:
a significant AUC decline since initial validation and one grade with far more defaults than its PD implies.

## An annual validation pass

[`examples/validation_report.py`](examples/validation_report.py), on a synthetic grade table:

```python
import numpy as np
import eqlx_credit as ec

n = np.array([820, 1450, 1910, 1320, 640, 210])  # obligors per grade at the start of the year
d = np.array([1, 4, 11, 19, 21, 17])  # defaults during the year
pd = np.array([0.0012, 0.0035, 0.0070, 0.0140, 0.0300, 0.0700])

cur = ec.auc_ci_from_counts(n, d)  # AUC with DeLong confidence interval
ec.auc_decline_test(0.80, cur)  # vs the AUC at initial validation
ec.jeffreys_test(n, d, pd).rows()  # one dict per grade, ready for a report table
ec.traffic_light(n, d, pd, rho=0.12).colours  # allowing for correlated defaults
ec.risk_weight(pd, 0.45)  # Basel IRB, corporate, LGD 45%, M 2.5
```

```text
AUC 0.785 (95% CI 0.734-0.836), Gini 0.570
Decline vs initial 0.800: S = 0.57, p = 0.285
grade 1:   1 defaults vs   1.0 expected, Jeffreys p = 0.421, traffic light green
grade 2:   4 defaults vs   5.1 expected, Jeffreys p = 0.662, traffic light green
grade 3:  11 defaults vs  13.4 expected, Jeffreys p = 0.734, traffic light green
grade 4:  19 defaults vs  18.5 expected, Jeffreys p = 0.437, traffic light green
grade 5:  21 defaults vs  19.2 expected, Jeffreys p = 0.327, traffic light green
grade 6:  17 defaults vs  14.7 expected, Jeffreys p = 0.259, traffic light green
Normal test over 5 years: z = 0.89
Risk weights: 33.0% 58.8% 80.5% 103.4% 128.4% 168.9%
```

## A low-default portfolio

[`examples/quickstart.py`](examples/quickstart.py):

```python
n_obligors = np.array([120, 340, 510, 430, 260, 90])
n_defaults = np.array([0, 0, 1, 2, 3, 2])

ec.pluto_tasche(n_obligors, n_defaults, confidence=0.9, rho=0.12)
ec.fit_ct_ar(n_obligors, central_tendency=0.015, accuracy_ratio=0.65).pd
ec.van_der_burgt(n_obligors, central_tendency=0.015, accuracy_ratio=0.65).pd
```

```text
Pluto-Tasche (90%, rho=0.12): [ 2.511  2.652  3.172  4.263  6.336 10.761] %
CT/AR curve:                  [ 0.037  0.135  0.421  1.205  3.383 10.692] %
van der Burgt:                [0.044 0.093 0.366 1.525 4.246 7.134] %
Master scale grades:          [ 2  5  8 11 14 17]
```

## How it is tested

- **Basel IRB:** all eight columns of Basel II Annex 5 ("Illustrative IRB Risk Weights") reproduced to
  the published two decimals.
- **Pluto–Tasche:** the paper's worked example (0.29%, 0.33%, 0.76%).
- **DeLong:** against an explicit obligor-by-obligor placement calculation, and by simulation: the
  standard error matches the spread of simulated AUCs and the 95% interval covers the true AUC about
  95% of the time.
- **Correlated binomial:** against a Monte Carlo simulation of the one-factor model.
- **Spiegelhalter:** a hand calculation, and standard normal under correct PDs by simulation.
- **Cohorts:** hand-built panels, including the horizon boundary and 29 February.

## Notes on the methods

- Use the obligor-average PD at the start of the period per grade as the grade PD in back-tests.
- `binomial_test` with `rho=0` assumes independent defaults and rejects too often when defaults are
  correlated; pass an asset correlation, or read it next to `traffic_light(..., rho=)`.
- Thresholds (test levels, traffic-light quantiles, PSI readings) are policy choices; set your own.
- `hosmer_lemeshow` uses `K` degrees of freedom for `K` grades (out-of-sample use).
- `pluto_tasche` is the single-period version; in practice the bounds are often rescaled to the central
  tendency afterwards.
- `fit_ct_ar` uses Tasche's robust-logit grade positions by default but matches the accuracy ratio
  exactly on the grade distribution rather than through his moment approximation;
  `placement="even"` gives the v0.1 behaviour.
- `van_der_burgt` returns each grade's PD as the average slope of the CAP curve over that grade, so the
  weighted mean equals the central tendency exactly; grading makes the grade-level AR a little lower.
- `risk_weight` defaults to no 1.06 scaling factor (Basel III final); pass `scaling=1.06` for Basel II /
  CRR. PD floors differ by framework and are the caller's choice.

## References

- Basel Committee on Banking Supervision (2006). International Convergence of Capital Measurement and
  Capital Standards, ¶272–330 and Annex 5.
- Basel Committee on Banking Supervision (2005). An Explanatory Note on the Basel II IRB Risk Weight
  Functions.
- Basel Committee on Banking Supervision (2005). Studies on the validation of internal rating systems.
  Working Paper No. 14.
- European Central Bank (2019). Instructions for reporting the validation results of internal models:
  IRB Pillar I models for credit risk.
- European Banking Authority (2017). Guidelines on PD estimation, LGD estimation and the treatment of
  defaulted exposures (EBA/GL/2017/16).
- DeLong, E. R., DeLong, D. M. and Clarke-Pearson, D. L. (1988). Comparing the areas under two or more
  correlated receiver operating characteristic curves. *Biometrics* 44(3).
- Spiegelhalter, D. J. (1986). Probabilistic prediction in patient management and clinical trials.
  *Statistics in Medicine* 5(5).
- Tasche, D. (2003). A traffic lights approach to PD validation. arXiv:cond-mat/0305038.
- Tasche, D. (2013). The art of probability-of-default curve calibration. *Journal of Credit Risk*
  9(4). arXiv:1212.3716.
- Pluto, K. and Tasche, D. (2005). Thinking positively. *Risk*, August 2005. Longer version:
  arXiv:cond-mat/0411699.
- van der Burgt, M. (2008). Calibrating low-default portfolios, using the cumulative accuracy profile.
  *Journal of Risk Model Validation* 1(4).
- Vasicek, O. (2002). The distribution of loan portfolio value. *Risk*, December 2002.
- Siddiqi, N. (2006). *Credit Risk Scorecards*. Wiley.

## Development

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check . && uv run mypy src
```

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
