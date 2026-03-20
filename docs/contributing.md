# Contributing Guide

Thank you for your interest in contributing!

We welcome bug reports, pull requests, and improvements of all kinds.

## Development Setup

Clone the repository and install in editable mode:

```bash
git clone https://github.com/transferable-samplers/transferable-samplers.git
cd transferable-samplers
uv venv .venv --python=3.11
source .venv/bin/activate
uv pip install -e .
uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt
```

### Enable pre-commit hooks

```bash
uv run pre-commit install
```

Hooks will now run automatically on every `git commit`. The pre-commit configuration enforces ruff formatting/linting and pyrefly type checking.

## Running Tools Manually

**Format + lint (auto-fix):**

```bash
uv run pre-commit run ruff --all-files
uv run pre-commit run ruff-format --all-files
```

**Type checking (Pyrefly):**

```bash
uv run pre-commit run pyrefly-check --all-files --hook-stage manual
```

**Run all pre-commit hooks on the entire repo:**

```bash
uv run pre-commit run --all-files
```

## Running Tests

This project uses **pytest** with Hydra integration. Tests are organised into marker categories:

| Marker | Description | Location |
|---|---|---|
| `essential` | Must-pass tests: config validation, model asset checks, and MWE pipeline tests | `tests/configs/`, `tests/assets/`, `tests/mwe/` |
| `benchmark` | Long-running experiments that validate metrics against published results | `tests/benchmark/` |
| `optional` | Architectural / design tests (e.g. NF invertibility, dlogp consistency) | `tests/unit/` |

### Essential tests

Essential tests cover config validation, model asset checks, and minimum working example (MWE) pipelines. GPU is the preferred way to run the full essential suite:

```bash
PYTEST_TRAINER=gpu uv run pytest -m essential -v
```

CPU can be used as a rapid sanity check or where a GPU is not available. ECNF++ SNIS MWE experiments are too slow on CPU and are automatically skipped.

```bash
PYTEST_TRAINER=cpu uv run pytest -m essential -v
```

DDP runs only the MWE tests (config and asset tests are skipped). This is useful for catching bugs specific to distributed execution.

```bash
PYTEST_TRAINER=ddp uv run pytest -m essential -v
```

### Benchmark tests

Benchmark tests run full experiment configs and validate metrics against reference values from stable codebase versions, as well as published paper results.

```bash
PYTEST_TRAINER=gpu uv run pytest -m benchmark -v
```

To check for performance degradation from incorrect distributed implementation:

```bash
PYTEST_TRAINER=ddp uv run pytest -m benchmark -v
```

### Optional tests

Architecture and design tests (e.g. NF invertibility, dlogp consistency):

```bash
uv run pytest -m optional -v
```

### Updating config snapshots

Config tests compare each Hydra experiment config against a reference JSON snapshot in `tests/configs/reference_configs/`. If you intentionally change a config, update the snapshots:

```bash
uv run pytest tests/configs/test_experiment_configs.py --update-reference
```

Then commit the updated JSON files alongside your config changes.
