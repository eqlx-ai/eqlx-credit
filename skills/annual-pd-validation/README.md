# Skill: annual PD validation

An Agent Skill that prepares the evidence for an annual PD model validation and drafts the report. The agent runs
the eqlx-credit tests through `scripts/run_validation.py`, reads the results, and writes an evidence draft. Every
number comes from the tool; the conclusion and signature stay with the validator.

- `SKILL.md`: the instructions the agent follows (steps, hard rules, report structure).
- `scripts/run_validation.py`: runs every test on one JSON input and writes `results.json`, with the function
  behind each result and the flags raised against the policy thresholds in the input.
- `examples/portfolio.json`: a synthetic portfolio with planted problems: a significant AUC decline since initial
  validation and one grade with far more defaults than its PD implies.

Use with Claude Code by copying this folder to `.claude/skills/annual-pd-validation/` in your project (or
`~/.claude/skills/`), then ask: "Yıllık PD validasyonunu yap: examples/portfolio.json".
