# eqlx-credit

Open tools for credit-risk model work: validation tests, calibration, master scales and
low-default portfolio methods for probability-of-default (PD) models.

Built by [eqlx](https://eqlx.ai) from public regulation and published research.
Part of [eqlx.org](https://eqlx.org), open credit-risk research.

> Status: alpha (0.1). The functions are tested against closed forms and published
> examples, but the API may still change.

## Install

```bash
pip install git+https://github.com/eqlx-ai/eqlx-credit
```

Python 3.10+, depends only on NumPy and SciPy.

## What is inside

| Area | Function | What it answers |
|---|---|---|
| Discrimination | `auc`, `gini`, `ks` | Does the model rank defaulters above non-defaulters? |
| Calibration tests | `binomial_test`, `jeffreys_test`, `hosmer_lemeshow` | Are the grade PDs consistent with observed defaults? |
| Stability | `psi`, `hhi` | Has the portfolio moved across grades? Is it concentrated? |
| Calibration | `anchor_to_central_tendency` | Shift PDs to a target average, keeping the ranking |
| Calibration | `fit_ct_ar`, `accuracy_ratio_from_grades` | Grade PDs from a target central tendency and accuracy ratio |
| Low-default portfolios | `pluto_tasche` | Most prudent PD upper bounds, independent or correlated defaults |
| Low-default portfolios | `van_der_burgt` | Grade PDs from a fitted CAP curve |
| Master scale | `geometric_master_scale`, `master_scale_from_midpoints` | The PD-to-grade ladder |

Conventions: grades run from lowest to highest risk; a higher score means higher risk;
`1` in a default flag means default.

## Quick start

```python
import numpy as np
import eqlx_credit as ec

# Obligors and defaults per grade, lowest risk first (synthetic).
n_obligors = np.array([120, 340, 510, 430, 260, 90])
n_defaults = np.array([0, 0, 1, 2, 3, 2])

pt = ec.pluto_tasche(n_obligors, n_defaults, confidence=0.9, rho=0.12)
fit = ec.fit_ct_ar(n_obligors, central_tendency=0.015, accuracy_ratio=0.65)
vdb = ec.van_der_burgt(n_obligors, central_tendency=0.015, accuracy_ratio=0.65)
scale = ec.geometric_master_scale(0.0003, 0.30, 20)
jt = ec.jeffreys_test(n_obligors, n_defaults, fit.pd)
```

Output of [`examples/quickstart.py`](examples/quickstart.py):

```text
Pluto-Tasche (90%, rho=0.12): [ 2.511  2.652  3.172  4.263  6.336 10.761] %
CT/AR curve:                  [ 0.043  0.132  0.401  1.209  3.592 10.185] %
van der Burgt:                [0.044 0.093 0.366 1.525 4.246 7.134] %
Master scale grades:          [ 2  5  8 11 14 17]
Jeffreys p-values:            [0.253 0.657 0.748 0.936 0.992 0.998]
Grades flagged:               []
Concentration (HHI):          0.055
```

## Notes on the methods

- **`jeffreys_test`** returns, per grade, the posterior probability (Jeffreys prior)
  that the true default rate is below the assigned PD. Small values mean the PD looks
  too low. This is the PD back-test in the ECB's instructions for IRB validation
  reporting.
- **`binomial_test`** assumes independent defaults and therefore rejects too often when
  defaults are correlated.
- **`hosmer_lemeshow`** uses `K` degrees of freedom for `K` grades (out-of-sample use).
- **`pluto_tasche`** is the single-period version. With `rho > 0` it integrates the
  one-factor Vasicek model numerically (Gauss-Hermite quadrature).
- **`fit_ct_ar`** follows the idea of Tasche's quasi-moment-matching approach (a
  two-parameter logistic PD curve set by a central tendency and an accuracy ratio),
  but matches the accuracy ratio exactly on the grade distribution rather than through
  a moment approximation. It is not a reproduction of the paper's estimator.
- **`van_der_burgt`** returns each grade's PD as the average slope of the CAP curve
  over that grade, so the weighted mean PD equals the central tendency exactly. The
  accuracy ratio of the continuous curve matches the target; grading coarsens it, so
  the grade-level accuracy ratio is a little lower.

## References

- Pluto, K. and Tasche, D. (2005). Thinking positively. *Risk*, August 2005.
  Longer version: "Estimating probabilities of default for low default portfolios", arXiv:cond-mat/0411699.
- van der Burgt, M. (2008). Calibrating low-default portfolios, using the cumulative
  accuracy profile. *Journal of Risk Model Validation* 1(4).
- Tasche, D. (2013). The art of probability-of-default curve calibration. *Journal of
  Credit Risk* 9(4). arXiv:1212.3716.
- Basel Committee on Banking Supervision (2005). Studies on the validation of internal
  rating systems. Working Paper No. 14.
- European Central Bank (2019). Instructions for reporting the validation results of
  internal models: IRB Pillar I models for credit risk.
- Vasicek, O. (2002). The distribution of loan portfolio value. *Risk*, December 2002.

## Development

```bash
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check . && uv run mypy src
```

## License

Apache-2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).
