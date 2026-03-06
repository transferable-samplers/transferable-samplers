# Contributing Guide

Thank you for your interest in contributing!

We welcome bug reports, pull requests, and improvements of all kinds.

Before contributing, please take a moment to read through this guide.

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

### Essential tests (CPU)

Essential tests cover config validation, model asset checks, and minimum working example (MWE) pipelines. Run them on CPU first:

```bash
PYTEST_TRAINER=cpu uv run pytest -m essential -v
```

To run the MWE pipeline tests on a single GPU or with DDP (requires 2+ GPUs):

```bash
PYTEST_TRAINER=gpu uv run pytest -m essential -v
PYTEST_TRAINER=ddp uv run pytest -m essential -v
```

Config and asset tests always run on CPU regardless of `PYTEST_TRAINER`. DDP mode only runs the MWE tests.

### Benchmark tests (GPU)

Benchmark tests run full experiment configs and validate metrics against published results. They require at least one GPU:

```bash
PYTEST_TRAINER=gpu uv run pytest -m benchmark -v
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
