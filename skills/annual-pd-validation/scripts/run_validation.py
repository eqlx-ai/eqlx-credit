"""Run every test of an annual PD validation on one input file and print the results as JSON.

Usage:  python run_validation.py portfolio.json [--out results.json]

Every number comes from eqlx-credit; this script only arranges inputs and outputs. Each result names the
function that produced it, so a report can cite it. Flags are raised only by comparing a result with a
threshold from the input's "policy" block; thresholds missing from the policy are reported as "not set".

Numbers are reported to 4 decimal places (the precision a validation report uses); flags are decided on the
unrounded values.
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import pairwise
from typing import Any

import numpy as np

import eqlx_credit as ec


def _need(data: dict[str, Any], key: str) -> Any:
    if key not in data:
        raise SystemExit(f"input is missing '{key}'")
    return data[key]


def _check_inputs(grades: list[dict[str, Any]]) -> list[str]:
    problems = []
    pds = [g["pd"] for g in grades]
    if any(b < a for a, b in pairwise(pds)):
        problems.append("grade PDs are not increasing: grades must run from lowest to highest risk")
    for g in grades:
        if g["defaults"] > g["obligors"]:
            problems.append(f"grade {g['grade']}: more defaults than obligors")
        if not 0 < g["pd"] < 1:
            problems.append(f"grade {g['grade']}: PD must lie strictly between 0 and 1")
    return problems


DECIMALS = 4


def _rounded(obj: Any) -> Any:
    if isinstance(obj, float):
        return round(obj, DECIMALS)
    if isinstance(obj, dict):
        return {k: _rounded(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_rounded(v) for v in obj]
    return obj


def run(data: dict[str, Any]) -> dict[str, Any]:
    """All results, rounded for reporting. Flags are computed before rounding."""
    return _rounded(_run(data))  # type: ignore[no-any-return]


def _run(data: dict[str, Any]) -> dict[str, Any]:
    grades = _need(data, "grades")
    problems = _check_inputs(grades)
    if problems:
        return {"input_problems": problems}
    names = [str(g["grade"]) for g in grades]
    n = np.array([g["obligors"] for g in grades], dtype=float)
    d = np.array([g["defaults"] for g in grades], dtype=float)
    pd = np.array([g["pd"] for g in grades], dtype=float)
    policy = data.get("policy", {})
    alpha = float(policy.get("alpha", 0.05))
    rho = policy.get("asset_correlation")
    flags: list[dict[str, str]] = []
    out: dict[str, Any] = {
        "portfolio": data.get("portfolio"),
        "observation_period": data.get("observation_period"),
        "eqlx_credit_version": ec.__version__,
        "summary": {
            "obligors": int(n.sum()),
            "defaults": int(d.sum()),
            "observed_default_rate": float(d.sum() / n.sum()),
            "obligor_weighted_pd": float(np.dot(n, pd) / n.sum()),
            "expected_defaults": float(np.dot(n, pd)),
        },
    }

    # 1. Discriminatory power
    cur = ec.auc_ci_from_counts(n, d)
    disc: dict[str, Any] = {
        "function": "auc_ci_from_counts",
        "auc": cur.auc,
        "auc_ci_95": [cur.ci_low, cur.ci_high],
        "gini": cur.gini,
        "gini_ci_95": list(cur.gini_ci),
        "std_error": cur.std_error,
    }
    if "initial_validation_auc" in data:
        t = ec.auc_decline_test(float(data["initial_validation_auc"]), cur, alpha=alpha)
        disc["decline_test"] = {
            "function": "auc_decline_test",
            "auc_initial": t.auc_initial,
            "statistic": t.statistic,
            "p_value": t.p_value,
            "significant_decline": t.rejected,
        }
        if t.rejected:
            flags.append(
                {
                    "area": "discrimination",
                    "code": "auc_decline",
                    "detail": (
                        f"AUC {t.auc_current:.3f} vs {t.auc_initial:.3f} at initial validation, "
                        f"p = {t.p_value:.4f}"
                    ),
                }
            )
    else:
        disc["decline_test"] = "not run: initial_validation_auc not given"
    out["discrimination"] = disc

    # 2. Calibration per grade
    jt = ec.jeffreys_test(n, d, pd, alpha=alpha)
    rows = jt.rows()
    for r, name in zip(rows, names, strict=True):
        r["grade"] = name
    calib: dict[str, Any] = {"per_grade": rows, "per_grade_function": "jeffreys_test"}
    for r in rows:
        if r["rejected"]:
            flags.append(
                {
                    "area": "calibration",
                    "code": "grade_pd_underestimated",
                    "detail": (
                        f"grade {r['grade']}: {r['defaults']} defaults vs "
                        f"{r['expected_defaults']:.1f} expected, Jeffreys p = {r['p_value']:.4f}"
                    ),
                }
            )
    thresholds = policy.get("traffic_light_thresholds")
    if thresholds:
        lights = {"independent": ec.traffic_light(n, d, pd, thresholds=tuple(thresholds)).colours}
        if rho is not None:
            lights[f"correlated_rho_{rho}"] = ec.traffic_light(
                n, d, pd, rho=float(rho), thresholds=tuple(thresholds)
            ).colours
        calib["traffic_light"] = {
            "function": "traffic_light",
            "thresholds": thresholds,
            "grades": names,
            **lights,
        }
        for view, colours in lights.items():
            for name, c in zip(names, colours, strict=True):
                if c != "green":
                    flags.append(
                        {
                            "area": "calibration",
                            "code": f"traffic_light_{c}",
                            "detail": f"grade {name}: {c} ({view})",
                        }
                    )
    else:
        calib["traffic_light"] = "not set: policy.traffic_light_thresholds"
    port = ec.jeffreys_test([n.sum()], [d.sum()], [float(np.dot(n, pd) / n.sum())], alpha=alpha)
    calib["portfolio_jeffreys"] = {
        "function": "jeffreys_test",
        "p_value": float(port.p_value[0]),
        "rejected": bool(port.rejected[0]),
    }
    if port.rejected[0]:
        flags.append(
            {
                "area": "calibration",
                "code": "portfolio_pd_underestimated",
                "detail": f"portfolio Jeffreys p = {float(port.p_value[0]):.4f}",
            }
        )
    hl = ec.hosmer_lemeshow(n, d, pd)
    calib["hosmer_lemeshow"] = {
        "function": "hosmer_lemeshow",
        "statistic": hl.statistic,
        "p_value": hl.p_value,
        "dof": hl.dof,
    }
    if hl.p_value < alpha:
        flags.append(
            {
                "area": "calibration",
                "code": "hosmer_lemeshow",
                "detail": f"Hosmer-Lemeshow p = {hl.p_value:.4f}",
            }
        )
    hist = data.get("history")
    if hist:
        nt = ec.normal_test(hist["default_rates"], hist["pds"])
        calib["normal_test"] = {
            "function": "normal_test",
            "years": hist.get("years"),
            "statistic": nt.statistic,
            "p_value": nt.p_value,
        }
        if nt.p_value < alpha:
            flags.append(
                {
                    "area": "calibration",
                    "code": "multi_year_underestimation",
                    "detail": f"normal test over {len(hist['default_rates'])} years, p = {nt.p_value:.4f}",
                }
            )
        calib["long_run_average_default_rate"] = {
            "function": "long_run_average",
            "value": ec.long_run_average(hist["default_rates"]),
        }
    else:
        calib["normal_test"] = "not run: history not given"
    out["calibration"] = calib

    # 3. Stability and concentration
    stab: dict[str, Any] = {"hhi": {"function": "hhi", "value": ec.hhi(n)}}
    if "previous_obligors" in data:
        value = ec.psi(data["previous_obligors"], n)
        stab["psi"] = {"function": "psi", "value": value}
        watch, material = policy.get("psi_watch"), policy.get("psi_material")
        if material is not None and value > material:
            flags.append(
                {"area": "stability", "code": "psi_material", "detail": f"PSI {value:.3f} > {material}"}
            )
        elif watch is not None and value > watch:
            flags.append({"area": "stability", "code": "psi_watch", "detail": f"PSI {value:.3f} > {watch}"})
    else:
        stab["psi"] = "not run: previous_obligors not given"
    out["stability"] = stab
    out["flags"] = flags
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("input")
    p.add_argument("--out")
    args = p.parse_args()
    with open(args.input, encoding="utf-8") as fh:
        result = run(json.load(fh))
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    print(text)
    return 1 if "input_problems" in result else 0


if __name__ == "__main__":
    sys.exit(main())
