# Contributing to eqlx-credit

Thanks for helping. Bug reports, new tests, documentation fixes and new methods are all welcome.

## Ground rules

1. **Public sources only.** Every method must come from published research or public regulation
   (BCBS, EBA, ECB, BDDK and similar). Add the reference to the list in `README.md`.
   Do not contribute code, text, thresholds or figures from confidential documents, vendor
   manuals or internal bank material.
2. **No real data.** Examples and tests use synthetic data only. Never add customer- or
   obligor-level data, even anonymised.
3. **Every function is tested.** A test must check the result against something independent:
   a closed form, a published worked example, or a brute-force calculation.
4. **Say what you implement.** If a function departs from the paper it cites, the docstring and
   README must say how.
5. **Dependencies:** NumPy and SciPy only. Propose anything else in an issue first.

Conventions: grades run from lowest to highest risk; a higher score means higher risk;
`1` in a default flag means default.

## Workflow

1. Open an issue first for anything larger than a small fix, so we can agree on scope.
2. Fork the repository and create a branch (`fix/...`, `feat/...`).
3. Run the checks locally:

   ```bash
   uv sync
   uv run pytest
   uv run ruff check . && uv run ruff format --check . && uv run mypy src
   ```

4. Open a pull request against `main` and fill in the template. CI must pass before review.
   For first-time contributors a maintainer approves the CI run.

By contributing you agree that your contribution is licensed under the Apache License 2.0,
the license of this project.

## Questions

Open an issue, or write to research@eqlx.ai.
