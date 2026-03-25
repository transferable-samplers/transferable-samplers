# Contributing Guide

Thank you for your interest in contributing!

We welcome bug reports, pull requests, and improvements of any kind.

## Contributing a Dataset or Method

Our hope is for this to become a community-driven project, and we'd be excited to hear from anyone interested in contributing a dataset or model/method.

We haven't finalised a process for this yet, but as a rule of thumb: any sampling-related dataset or method with published results (preprints and workshop papers are fine!) is a good candidate. If you have something in mind, please open an issue or reach out, and we'd be happy to discuss.

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

### Benchmark tests

Benchmark tests run full experiment configs and validate metrics against reference values from stable codebase versions, as well as published paper results.

```bash
PYTEST_TRAINER=gpu uv run pytest -m benchmark -v
```

### DDP Benchmarks

Pytest was not found to work nicely with DDP, hence DDP tests must be run manually. The following commands validate distributed execution; batch sizes are tuned for 4×48 GB GPUs:

```bash
# tests/benchmark/test_snis_prose_up_to_8aa.py::test_snis_prose_up_to_8aa
python -m transferable_samplers.eval \
  experiment=transferable/eval/prose_up_to_8aa_snis.yaml \
  trainer=ddp \
  data.test_sequences=ARIP \
  callbacks.sampling_evaluation.sampler.num_samples=10_000 \
  model.source_energy_config.sample_batch_size=4_096

# tests/benchmark/test_self_improve_prose_up_to_8aa.py::test_self_improve_prose_up_to_8aa
python -m transferable_samplers.train \
  experiment=transferable/finetune/prose_up_to_8aa_self_improve.yaml \
  trainer=ddp \
  data.test_sequences=ARIP \
  model.source_energy_config.sample_batch_size=4_096

# tests/benchmark/test_snis_ecnf_up_to_4aa.py::test_snis_ecnf_up_to_4aa
python -m transferable_samplers.eval \
  experiment=transferable/eval/ecnf++_up_to_4aa_snis.yaml \
  trainer=ddp \
  data.test_sequences=AA \
  callbacks.sampling_evaluation.sampler.num_samples=10_000 \
  model.source_energy_config.sample_batch_size=512

# tests/benchmark/test_smc_prose_up_to_8aa.py::test_smc_prose_up_to_8aa
python -m transferable_samplers.eval \
  experiment=transferable/eval/prose_up_to_8aa_mala.yaml \
  trainer=ddp \
  data.test_sequences=ARIP \
  callbacks.sampling_evaluation.sampler.num_samples=10_000 \
  model.source_energy_config.sample_batch_size=4_096 \
  model.source_energy_config.energy_batch_size=384 \
  model.source_energy_config.grad_batch_size=384
```

### Optional tests

Architecture and design tests (e.g. NF invertibility, dlogp consistency):

```bash
uv run pytest -m optional -v
```

### Test coverage

Run the essential suite with branch coverage:

```bash
PYTEST_TRAINER=cpu uv run pytest -m essential --cov --cov-report=html --cov-report=xml -v
```

Open the HTML report after the run:

```bash
open htmlcov/index.html
```

### Updating config snapshots

Config tests compare each Hydra experiment config against a reference JSON snapshot in `tests/configs/reference_configs/`. If you intentionally change a config, update the snapshots:

```bash
uv run pytest tests/configs/test_experiment_configs.py --update-reference
```

Then commit the updated JSON files alongside your config changes.
