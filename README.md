# CARR-v2

[![CI](https://github.com/jkinarthur/carr-opensource/actions/workflows/ci.yml/badge.svg)](https://github.com/jkinarthur/carr-opensource/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Differentiable collapse-aware compression for LLM-based recommendation, including:

- Unified differentiable compression loss.
- Learnable loss-weight adaptation via bilevel optimization.
- Gumbel-softmax depth selection.
- Drift-aware diagnostics, tracking, and plotting utilities.

## Highlights

- Research-oriented PyTorch implementation in src/carr_v2.
- Reproducible scripts and examples for synthetic and real datasets.
- Manuscript and supplementary material under paper.
- Governance, CI, and contribution standards aligned with modern open-source practice.

## Repository Layout

- src/carr_v2: Core library modules.
- examples: Runnable demos and experiment entry points.
- scripts: Utility scripts for dataset prep, remote orchestration, and audits.
- outputs: Generated experiment artifacts.
- paper: Manuscript and supplementary tex sources.

## Quick Start

### 1) Environment

Python 3.10+ is required.

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -U pip
pip install -e .[dev]
```

### 2) Sanity Run

```bash
python examples/mini_trainer.py
python examples/train_step_demo.py
```

### 3) Lint and Test

```bash
python -m compileall -q src
ruff check src tests scripts
ruff format --check src tests scripts
pytest
```

## Reproducibility Notes

- Seed all runs for deterministic comparisons where possible.
- Track hardware constraints and failure modes explicitly in outputs and manuscript tables.
- Keep large binary artifacts out of git unless needed for a specific release.

## Documentation and Policy

- Contributing guidelines: CONTRIBUTING.md
- Security policy: SECURITY.md
- Code of conduct: CODE_OF_CONDUCT.md
- Changelog: CHANGELOG.md

## Citation

If you use this repository in research, see CITATION.cff for preferred citation metadata.

## License

This project is licensed under the MIT License. See LICENSE.
