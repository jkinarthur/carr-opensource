# Contributing

Thank you for contributing to CARR-v2.

## Development Workflow

1. Fork and create a feature branch.
2. Install development dependencies:

```bash
pip install -e .[dev]
pre-commit install
```

3. Run checks before opening a PR:

```bash
python -m compileall -q src
ruff check tests
ruff format --check tests
pytest
```

4. Open a pull request with a clear description, test notes, and scope.

## Coding Standards

- Target Python 3.10+.
- Follow Ruff linting and formatting rules in pyproject.toml.
- Prefer small, focused commits.
- Add or update tests when behavior changes.

## Commit Message Guidance

Use short, imperative messages. Example:

- Add leakage-audit status validation
- Fix dataset loader shape mismatch

## Pull Request Checklist

- Tests pass locally.
- Lint and formatting checks pass.
- Documentation is updated where applicable.
- Changelog entry is added for user-facing changes.

## Research Artifacts

Large generated files should not be committed unless required for reproducibility of a release or paper revision.
