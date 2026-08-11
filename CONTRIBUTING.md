# Contributing

1. Create a branch from `main`.
2. Install development dependencies with `pip install -e ".[dev]"`.
3. Add focused tests for behavioral and security changes.
4. Run `ruff check .`, `bandit -r app -q`, `pytest --cov=app`, `pip-audit --local --skip-editable`, and `python -m scripts.run_rag_eval`.
5. Open a pull request describing the behavior, risk, and validation performed.

Report security-sensitive findings through the process in [SECURITY.md](SECURITY.md).
