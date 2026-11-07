---
name: annual-pd-validation
description: Prepare the evidence for an annual validation of a PD (probability of default) rating model from a grade table — discriminatory power, calibration per grade and across years, stability — and draft the validation report. Use when asked to validate, back-test or review a PD model or rating system for a period. Every number comes from the eqlx-credit tool; the agent never computes or invents figures and never gives the validation opinion.
---

# Annual PD validation

You prepare the **evidence** for an annual PD model validation and draft the report. A named validator reviews it
and gives the opinion. Your job is to run the right tests in the right order, read their results correctly, and
write them up so the validator can check every figure.

## Hard rules

1. **Never compute, estimate, round differently or invent a number.** Every figure in the report must appear in
   the tool output (`results.json`), which already gives numbers to 4 decimals; copy them as they are (you may show
   a rate as a percentage only if you keep the same digits, e.g. 0.0531 as 5.31%). If a figure is not there, write "not computed" and say what input is missing.
2. **Thresholds are the institution's policy.** Use only the values in the input's `policy` block. If one is
   missing, say "not set" in the report and ask the user for it; do not pick one yourself.
3. **No validation opinion.** Do not write "the model is valid", "approved", "fit for purpose" or similar. Report
   findings and open questions. The conclusion and signature lines stay empty for the validator.
4. **No customer data in the report** beyond the aggregated grade table.
5. If the tool reports `input_problems`, stop, show them, and ask the user to fix the input.

## Steps

### 1. Get the input
Ask for, or locate, one JSON file in this shape (see `examples/portfolio.json`):

- `grades`: per grade, lowest risk first: `grade`, `obligors` (performing at the start of the period), `defaults`
  (during the period), `pd` (the obligor-average PD of the grade at the start of the period).
- `initial_validation_auc` (optional): the AUC at initial validation, for the decline test.
- `previous_obligors` (optional): obligors per grade a year earlier, for the stability index.
- `history` (optional): yearly portfolio `default_rates` and `pds`, for the multi-year test.
- `policy`: `alpha`, `asset_correlation`, `traffic_light_thresholds`, `psi_watch`, `psi_material`.

Before running anything, show the user a short summary of the input (grades, obligors, defaults, which optional
parts are present) and confirm it is the right portfolio and period.

### 2. Run the tool

```bash
uv run --with eqlx-credit \
  python scripts/run_validation.py <input.json> --out results.json
```

(or `python scripts/run_validation.py ...` where `eqlx-credit` is installed). Paths are relative to this skill's
folder. The script runs every test with eqlx-credit and writes all results, the function behind each one, and the
`flags` raised against the policy thresholds.

### 3. Read the results
Go through `results.json` in this order and note what each part says:

- **Discrimination:** AUC and Gini with their 95% intervals. If `decline_test.significant_decline` is true, this
  is a lead finding: the model ranks worse than at initial validation.
- **Calibration per grade:** for each grade, defaults against expected defaults and the Jeffreys p-value. A
  rejected grade means more defaults than its PD implies: the PD looks too low.
- **Traffic lights:** compare the independent and the correlated view. If a grade is yellow or red only under
  independence, say so: correlated defaults in a bad year can explain part of the excess, which the validator
  must judge.
- **Portfolio level:** the portfolio Jeffreys test and Hosmer-Lemeshow.
- **Across years:** the normal test and the long-run average default rate against the PDs.
- **Stability:** PSI against the previous distribution, HHI concentration.

### 4. Draft the report
Write it in the user's language (Turkish if the user writes Turkish), in this structure:

1. **Scope**: portfolio, observation period, data used, eqlx-credit version.
2. **Summary of findings**: the `flags`, most serious first (discrimination decline and rejected grades before
   warnings), one line each with the figure that supports it.
3. **Discriminatory power**: a table of AUC, Gini, intervals, decline test.
4. **Calibration**: a per-grade table (grade, obligors, defaults, expected defaults, observed rate, PD, Jeffreys
   p-value, traffic light independent / correlated), then portfolio and multi-year results.
5. **Stability**: PSI and HHI.
6. **Open questions for the validator**: what the evidence cannot settle, for example whether a grade's excess
   defaults reflect a bad year or a mis-calibrated PD, or whether the decline in AUC calls for redevelopment.
7. **Method and sources**: list the eqlx-credit functions used (from `results.json`) and their public sources:
   ECB instructions for reporting IRB validation results (AUC decline, Jeffreys), BCBS Working Paper 14 (normal
   test, correlated binomial), Tasche (2003) traffic lights, DeLong et al. (1988), EBA/GL/2017/16 (long-run average).
8. **Conclusion and sign-off**: leave empty, with a line for the validator's name, date and opinion.

Mark the document as **"Evidence draft — not a validation opinion"**.

### 5. Check before handing over
Read your draft against `results.json`: every number in the report must match the tool output exactly. Then give
the user the report and the path to `results.json`.
