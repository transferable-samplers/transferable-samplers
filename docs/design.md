# Design

This page describes how the major components of the codebase fit together: the data module, Lightning module, samplers, and callbacks. The goal is to make the responsibilities of each component clear and explain the design decisions behind the boundaries between them.

?> Transferable samplers builds upon [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/).
A major benefit of PyTorch Lightning is in standardising the components of codebases, aiding both readability and reproducibility.
As such, we have sought to follow Lightning's preferred design principles where possible.
However, the algorithms implemented by transferable samplers do deviate from the typical workflows for which PyTorch Lightning was developed.
Care was given to deciding the correct boundaries between components in these non-standard configurations, with the hope that the result can be readily read, understood, hacked, and extended.

?> Whilst attention has been given to defining a general architecture, changes may be made as required by future methods.

## Overview

To the greatest practical extent we have aimed for **each component to own only a single concern**.

Data preparation and normalisation are handled by the data module.
The Lightning module defines how to train a generative model, exposing a source / proposal density for samplers to leverage.
Sampling strategy lives in the sampler, agnostic to the implementations of the source and target densities themselves.
This separation makes it possible to change any one component — swap a sampler, change a dataset, or try a new proposal model — without touching the others.

---

## Data Module

**Responsibility:** preparing, loading, and normalising data; providing evaluation contexts.

`BaseDataModule` is an abstract `LightningDataModule` with three required methods:

- `prepare_data()` — download and preprocess (runs in a single process; no state assignment).
- `setup()` — load datasets and set `self.data_train`; called before any fit/val/test stage. `self.data_val` and `self.data_test` are placeholders - evaluation is handled via callbacks.
- `prepare_eval(sequence, stage) -> EvalContext` — build a self-contained evaluation context for a given peptide sequence.

The two concrete subclasses — `SinglePeptideDataModule` and `ManyPeptidesDataModule` — only implement these three methods. All dataloader machinery is inherited from the base class.

?> The placeholder val and test dataloaders exist only to satisfy Lightning's requirements. Real evaluation does not happen through the dataloader loop — it happens through callbacks that call `prepare_eval()` directly. This is intentional, and allows metrics to be recorded for individual systems before aggregation.

### EvalContext

`EvalContext` bundles everything needed to evaluate a single peptide system:

- `true_data` — reference conformations (`SamplesData`) for metric computation.
- `target_energy` — energy function for use by samplers and evaluators, provides `energy()` and `energy_and_grad()` for the forcefield.
- `normalisation_std` — so evaluators can destandardise to physical units.
- `system_cond` — encoding / permutation tensors for transferable models.
- `tica_model`, `topology` — optional metadata for TICA-based metrics and chirality detection.

!> The data module handles standardisation. The only other component that sees physical-scale coordinates is the evaluator; neither the Lightning module nor the sampler operate on physical-scale data. `TargetEnergy` handles unnormalisation internally — callers always pass normalised coordinates.

---

## Lightning Module

**Responsibility:** defining the generative model, computing losses, and providing proposal / source energy for samplers.

`BaseLightningModule` is deliberately passive. It defines computation hooks and manages state (optimizer, buffer, EMA access), but it does not decide when to sample, when to evaluate, or how to orchestrate the training loop; that responsibility is delegated to callbacks.

Subclasses implement four abstract methods:

- `compute_primary_loss(batch)` — per-sample primary loss, without system-size normalisation or auxiliary terms. Exposed directly so that `LossEvaluationCallback` can evaluate model fit on held-out true samples without training-specific overhead.
- `training_step(batch, batch_idx)` — calls `compute_primary_loss` and adds system-size normalisation and any auxiliary terms to produce the final scalar loss.
- `generate_proposal(net, num_samples, system_cond)` — draw samples from the proposal and return `(samples, E_source)` where `E_source = -log q`.
- `proposal_energy(net, x, system_cond)` — evaluate `-log q` for given conformations.

Both `NormalizingFlowLitModule` and `FlowMatchLitModule` implement these four methods with identical signatures and semantics, so samplers and evaluators treat them interchangeably.

?> `validation_step` and `test_step` exist only for Lightning compatibility. All evaluation is callback-driven.

?> Both model types follow the same change-of-variables convention: `logq = logp_prior + dlogp`, where `dlogp` is the log-determinant of the Jacobian (NF) or the trace integral along the ODE trajectory (FM). Energy is uniformly `E = -logq`. For reverse-direction computations, both models consistently subtract the reverse-direction `dlogp`.

### SourceEnergy

`SourceEnergy` is the interface between the Lightning module and the samplers. `build_source_energy()` constructs one by wrapping `generate_proposal` and `proposal_energy` into a self-contained dataclass that handles internal batching, optional centre-of-mass energy adjustments, and DDP batch-size scaling. Samplers receive only this dataclass — they never see the model itself. `SourceEnergy` exposes three operations:

- `sample(num_samples)` — draw proposals in internal batches.
- `energy(x)` — evaluate `-log q` in internal batches.
- `energy_and_grad(x)` — evaluate `-log q` and its gradient in internal batches.


### Replay buffer

For self-improvement training, the model owns a `Buffer` that stores resampled conformations in normalised space. The model does not populate its own buffer — it only exposes `set_buffer()` and draws from `self._buffer.sample()` in `training_step` when `train_from_buffer=True`. The `PopulateBufferCallback` is responsible for filling it.

!> For consistency with the true data pipeline the buffer samples are destandardised before passing to the datamodule train transforms. This breaks the "rule" of the Lightning module not seeing physical scale coordinates but is accepted as it occurs inside the buffer and enables the exact same transform pipeline to be applied to buffer samples.

---

## Samplers

**Responsibility:** generating conformations from a proposal distribution and reweighting / annealing them toward a target energy.

Samplers receive `SourceEnergy` and `TargetEnergy` and return `(samples_dict, diagnostics)`:

```python
samples_dict, diagnostics = sampler.sample(source_energy, target_energy)
```

`samples_dict` maps names (e.g. `"resampled"`, `"smc"`) to `SamplesData` instances (as expected by the evaluator). `diagnostics` is sampler-specific output (e.g. SMC particle trajectory) or `None`.

?> Samplers own DDP coordination, as particles must be gathered between devices for resampling, and resharded to devices in the case of SMC.

The two concrete samplers:

- **`SNISSampler`** — self-normalised importance sampling. Draws proposals from `source_energy.sample()`, evaluates target and source energies, computes log importance weights `logw = -E_target + E_source`, and resamples proportional to the weights.
- **`SMCSampler`** — sequential Monte Carlo over an annealing path from source to target. Particles are sharded across ranks; MCMC runs locally per rank; resampling coordinates via `all_gather`.

---

## Callbacks

**Responsibility:** orchestrating sampling, evaluation, and buffer population.

Callbacks connect the otherwise-decoupled components. Because Lightning passes both `trainer` and `pl_module` to every callback hook, callbacks have access to the data module, the model, and the trainer state simultaneously.

### LossEvaluationCallback

Runs on `on_validation_epoch_end`. Calls `pl_module.compute_primary_loss()` on held-out true samples from the data module. This measures model fit separately from the sampling-based evaluation, without requiring a full sampling pass. By separating this from the built-in Lightning evaluation we trivially compute values per-system to enable more informative validation.

### SamplingEvaluationCallback

Runs on `on_validation_epoch_end` and `on_test_epoch_end`. For each sequence in `val_sequences` / `test_sequences`:

1. Calls `datamodule.prepare_eval(sequence, stage)` to get an `EvalContext`.
2. Calls `pl_module.build_source_energy(eval_ctx.system_cond)` to get a `SourceEnergy`.
3. Calls `sampler.sample(source_energy, target_energy)` — all ranks participate.
4. On rank zero: passes results to `PeptideEnsembleEvaluator.evaluate()`, logs metrics and plots.

### PopulateBufferCallback

Runs on `on_train_epoch_start` for self-improvement training. Uses the same sampling pattern as `SamplingEvaluationCallback`, but additionally creates a `Buffer` and calls `pl_module.set_buffer()`.

### EMAWeightAveraging

Maintains an exponential moving average of model weights. When present, `build_source_energy(use_ema_if_available=True)` uses the EMA weights in the `SourceEnergy` passed to the sampler.

!> `EMAWeightAveraging` is incompatible with `PopulateBufferCallback`.
