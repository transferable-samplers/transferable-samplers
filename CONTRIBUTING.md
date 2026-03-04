# Contributing Guide

Thank you for your interest in contributing 🌟

We welcome bug reports, pull requests, and improvements of all kinds.

Before contributing, please take a moment to read through this guide.

## 📦 Development Setup

Clone the repository and install the development environment:

```bash
git clone https://github.com/transferable-samplers/transferable-samplers.git
cd transferable-samplers
uv pip install -r requirements-dev.txt
```

### Enable pre-commit hooks

```bash
uv pip install pre-commit
pre-commit install
```

Hooks will now run automatically on every `git commit`.

## Running Tools Manually

**Format + lint (auto-fix):**

```bash
ruff format .
ruff check --fix .
```

**Type checking (Pyrefly):**

```bash
# Via pre-commit (manual stage):
pre-commit run pyrefly-check --all-files

# Or directly:
pyrefly check .
```

**Run all pre-commit hooks on the entire repo:**

```bash
pre-commit run --all-files
```

## Running Tests

This project uses **pytest** with Hydra integration. Tests are organised into three marker categories:

| Marker | Description | Location |
|---|---|---|
| `essential` | Must-pass tests: config validation, model asset checks, and MWE pipeline tests | `tests/configs/`, `tests/assets/`, `tests/mwe/` |
| `benchmark` | Long-running runs that validate metrics against the paper (GPU) | `tests/benchmark/` |
| `benchmark_ddp` | Same as `benchmark` but under DDP to verify parity with single-GPU | `tests/benchmark/` |
| `optional` | Architectural / design tests (e.g. NF invertibility, dlogp consistency) | `tests/unit/` |

### Hardware modes

Config tests (`tests/configs/`) and asset tests (`tests/assets/`) are **CPU-only**.

MWE pipeline tests (`tests/mwe/`) run on the trainer specified by the `PYTEST_TRAINER` environment variable (defaults to `gpu`):

```bash
PYTEST_TRAINER=cpu pytest -m essential -v   # CPU
PYTEST_TRAINER=gpu pytest -m essential -v   # single GPU (default)
PYTEST_TRAINER=ddp pytest -m essential -v   # DDP (requires 2+ GPUs)
```

### Quick reference

**Run all essential tests (config + assets + MWE) on GPU:**

```bash
pytest -m essential -v
```

**Run only config drift tests (CPU):**

```bash
pytest tests/configs/ -v
```

**Run only asset / model-weight tests (CPU):**

```bash
pytest tests/assets/ -v
```

**Run benchmark tests on GPU (full experiment configs):**

```bash
PYTEST_TRAINER=gpu pytest -m benchmark -v
```

**Run benchmark DDP parity tests (requires 2+ GPUs):**

```bash
PYTEST_TRAINER=ddp pytest -m benchmark_ddp -v
```

**Run optional architecture tests:**

```bash
pytest -m optional -v
```

### Config drift tests

Config tests verify that Hydra experiment configs haven't changed unexpectedly. They compare each composed config against a reference JSON snapshot in `tests/configs/reference_configs/` and check that all `_target_` paths are importable.

If you intentionally change a config (e.g., add a new callback or modify defaults), update the reference files:

```bash
pytest tests/configs/test_experiment_configs.py --update-reference
```

Then commit the updated JSON snapshots alongside your config changes.

### Testing with SLURM

For convenience we provide SLURM scripts for background testing on SLURM-enabled clusters:

```bash
sbatch tests/slurm/test_integrations_gpu.sh
sbatch tests/slurm/test_integrations_ddp.sh
```

These will generate report files that can be merged as:

```bash
junitparser merge tests/reports/report_ddp.xml tests/reports/report_gpu.xml tests/reports/report.xml
```

And then converted to html:

```bash
junit2html tests/reports/report.xml tests/reports/report.html
```
