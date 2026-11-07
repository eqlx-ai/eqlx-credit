import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

import eqlx_credit as ec

SKILL = Path(__file__).parents[1] / "skills" / "annual-pd-validation"
spec = importlib.util.spec_from_file_location("run_validation", SKILL / "scripts" / "run_validation.py")
rv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rv)
EXAMPLE = json.loads((SKILL / "examples" / "portfolio.json").read_text())


def test_example_raises_the_planted_findings():
    out = rv.run(EXAMPLE)
    codes = [f["code"] for f in out["flags"]]
    assert codes == ["auc_decline", "grade_pd_underestimated", "traffic_light_yellow", "hosmer_lemeshow"]
    assert "grade 5" in out["flags"][1]["detail"]


def test_numbers_are_exactly_the_library_results():
    out = rv.run(EXAMPLE)
    g = EXAMPLE["grades"]
    n = np.array([x["obligors"] for x in g])
    d = np.array([x["defaults"] for x in g])
    pd = np.array([x["pd"] for x in g])
    cur = ec.auc_ci_from_counts(n, d)
    r4 = lambda x: round(float(x), 4)  # noqa: E731 - the script reports 4 decimals
    assert out["discrimination"]["auc"] == r4(cur.auc)
    assert out["discrimination"]["decline_test"]["p_value"] == r4(ec.auc_decline_test(0.86, cur).p_value)
    assert [r["p_value"] for r in out["calibration"]["per_grade"]] == [
        r4(x) for x in ec.jeffreys_test(n, d, pd).p_value
    ]
    assert out["stability"]["psi"]["value"] == r4(ec.psi(EXAMPLE["previous_obligors"], n))
    tl = out["calibration"]["traffic_light"]
    assert tl["independent"][4] == "yellow" and tl["correlated_rho_0.12"][4] == "green"


def test_missing_optional_parts_and_policy_are_reported_not_invented():
    data = {"grades": EXAMPLE["grades"]}
    out = rv.run(data)
    assert out["discrimination"]["decline_test"].startswith("not run")
    assert out["calibration"]["traffic_light"].startswith("not set")
    assert out["calibration"]["normal_test"].startswith("not run")
    assert out["stability"]["psi"].startswith("not run")
    assert all(f["code"] != "psi_watch" for f in out["flags"])


def test_bad_input_stops_the_run():
    data = copy.deepcopy(EXAMPLE)
    data["grades"][2]["pd"] = 0.5  # breaks the low-to-high risk order
    data["grades"][0]["defaults"] = 9999
    out = rv.run(data)
    assert list(out) == ["input_problems"] and len(out["input_problems"]) == 2


def test_cli_writes_json(tmp_path, capsys, monkeypatch):
    target = tmp_path / "results.json"
    monkeypatch.setattr(
        "sys.argv", ["run_validation.py", str(SKILL / "examples" / "portfolio.json"), "--out", str(target)]
    )
    assert rv.main() == 0
    assert json.loads(target.read_text())["flags"]
    assert json.loads(capsys.readouterr().out)["eqlx_credit_version"] == ec.__version__


def test_skill_file_has_the_required_frontmatter():
    text = (SKILL / "SKILL.md").read_text()
    head = text.split("---")[1]
    assert "name: annual-pd-validation" in head and "description:" in head
    assert "Never compute" in text and "No validation opinion" in text


@pytest.mark.parametrize("key", ["grades"])
def test_required_key(key):
    with pytest.raises(SystemExit):
        rv.run({})
