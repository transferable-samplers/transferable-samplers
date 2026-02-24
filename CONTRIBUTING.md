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

This project uses **pytest** with Hydra integration. Currently the tests all require at least a single GPU, due to OpenMM + CPU issues.

### Run GPU tests

```bash
PYTEST_TRAINER=gpu pytest -v
```

### Run DDP tests (requires 2+ GPUs)

```bash
PYTEST_TRAINER=ddp pytest -v
```

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
