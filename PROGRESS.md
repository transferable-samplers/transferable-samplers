# Refactoring Progress

Tracking progress against the plan in REFACTOR.md.

## Phase 1: Extract utility functions — DONE

### 1a. `src/utils/chirality.py` (new file)
Moved from `src/models/utils.py`:
- `create_adjacency_list`, `get_atom_types`, `get_adj_list`
- `find_chirality_centers`, `compute_chirality_sign`
- `check_symmetry_change`, `get_symmetry_change`

Updated `transferable_boltzmann_generator_module.py` to import directly from `src.utils.chirality`.
Removed re-exports from `src/models/utils.py`.

### 1b. `src/utils/resampling.py` (new file)
- `resample_multinomial(x, logw)` — extracted from `src/models/utils.py`
- `resample_systematic(x, logw)` — extracted from `SMCSampler.resample` in `base_sampler.py`
- `com_energy_adjustment(x, com_std)` — extracted from `NormalizingFlowLitModule.com_energy_adjustment`

Updated callers:
- `transferable_boltzmann_generator_module.py` imports `resample_multinomial` from `src.utils.resampling`
- `SMCSampler.resample` in `base_sampler.py` delegates to `resample_multinomial`/`resample_systematic`

Removed the old `resample()` function and re-exports from `src/models/utils.py`.

### 1c. `src/utils/dataclasses.py` (new file)
- `SamplesData` — moved from `src/utils/data_types.py`
- `ProposalCond` — new dataclass with `permutations` and `encodings` fields
- `EvalContext` — new dataclass with `true_samples`, `target_energy_fn`, `proposal_cond` (optional), `tica_model`, `topology`

Deleted `src/utils/data_types.py`. Updated all imports to use `src.utils.dataclasses` directly.

### Phase 4 (partial): `prepare_eval` returns `EvalContext`
Done early since it depended only on Phase 1c.

- `SinglePeptideDataModule.prepare_eval` — returns `EvalContext` with `proposal_cond=None`, renamed param `prefix` → `stage`
- `TransferablePeptideDataModule.prepare_eval` — returns `EvalContext` with `ProposalCond(permutations=..., encodings=...)`, renamed param `prefix` → `stage`, added assertion that stage matches sequence subset
- Updated all 3 call sites:
  - `evaluate_all()` — unpacks from `eval_ctx`, handles `proposal_cond=None`
  - `generate_and_resample()` — unpacks from `eval_ctx`, passes `stage="test"`
  - `NormalizingFlowLitModule.model_step()` — fixed pre-existing bug (3-value unpack of 5-tuple), now caches `self.eval_ctx`

## Phase 2: Extract priors to package — DONE

Replaced single file `src/models/priors.py` with `src/models/priors/` package:
- `git rm src/models/priors.py`
- Created `src/models/priors/prior.py` — abstract `Prior` base class with `sample()` and `energy()` abstract methods
- Created `src/models/priors/normal_distribution.py` — `NormalDistribution(Prior)` with full existing implementation (without `__main__` test block)
- Created `src/models/priors/__init__.py` — empty (package marker only, no re-exports)
- Updated `transferable_boltzmann_generator_module.py` import to `from src.models.priors.normal_distribution import NormalDistribution`

Verified: `test_train_mwe[cpu-ecnf++_AAA]` and `test_train_mwe[cpu-tarflow_AAA]` pass.

## Phase 3: Extract MCMC kernels — DONE

Extracted MCMC kernel logic into free functions in `src/models/samplers/mcmc/`:
- `src/models/samplers/mcmc/ula.py` — `ula_kernel()` (ULA step, no MH acceptance)
- `src/models/samplers/mcmc/mala.py` — `mala_kernel()` (MALA step with MH acceptance)
- `src/models/samplers/mcmc/__init__.py` — re-exports both kernels

Removed HMC entirely (no published results):
- Deleted `hmc_sampler.py` and did not create `mcmc/hmc.py`

Consolidated `SMCSamplerULA`, `SMCSamplerMALA` into a single `SMCSampler` class:
- Added `kernel_type` parameter (`"ula"`, `"mala"`) to `SMCSampler.__init__`
- `mcmc_kernel()` dispatches to the appropriate kernel function
- Unified `linear_energy_interpolation_gradients` to always return `(x_grad, t_grad)` — both kernels receive the same signature, MALA discards `t_grad`
- Deleted `ula_sampler.py`, `mala_sampler.py`, `hmc_sampler.py`

Updated all config YAMLs:
- `_target_: src.models.samplers.ula_sampler.SMCSamplerULA` → `_target_: src.models.samplers.base_sampler.SMCSampler` + `kernel_type: ula`
- `_target_: src.models.samplers.mala_sampler.SMCSamplerMALA` → `_target_: src.models.samplers.base_sampler.SMCSampler` + `kernel_type: mala`

Verified: `test_smc_mwe[cpu-tarflow_AAA_ula]` passes.

## Remaining Phases

- **Phase 4**: ~~Unify datamodule `prepare_eval`~~ (done as part of Phase 1)
- **Phase 5**: Create sampling strategies (`BaseSamplingStrategy`, `SNISStrategy`, `SMCStrategy`)
- **Phase 6**: Create evaluator and callback
- **Phase 7**: Migrate Lightning modules
- **Phase 8**: Cleanup

# Phase 5: Create sampling strategies — DONE

Created three new files alongside existing code (no modifications to existing files):

### 5a. `src/models/samplers/base_sampling_strategy.py`
- `BaseSampler(ABC)` — abstract base class
- `sample()` — abstract method returning `dict[str, SamplesData]`
- `sample_proposal_in_batches(model, num_samples, proposal_cond)` — generates proposals in batches on local rank, handles DDP `all_gather`
- Accepts `ProposalCond` from `src.utils.dataclasses`
- Uses `TYPE_CHECKING` guard for `LightningModule` to avoid circular imports

### 5b. `src/models/samplers/snis_sampler.py`
- `SNISSampler(BaseSampler)` — Self-Normalized Importance Sampling
- Generates proposals via `sample_proposal_in_batches`
- Computes target energy, importance weights (`logits = -target_energy - log_q`)
- Optional CoM energy adjustment (`use_com_adjustment` flag, uses `com_energy_adjustment` from `src.utils.resampling`)
- Optional logit clipping (`logit_clip_filter`)
- Multinomial resampling via `resample_multinomial`
- Returns `{"proposal": SamplesData, "resampled": SamplesData}`

### 5c. `src/models/samplers/smc_sampler.py`
- `SMCSampler(BaseSampler)` — Sequential Monte Carlo
- First runs SNIS (inline, same logic as `SNISSampler`) to get proposal + resampled results
- Then runs full SMC loop extracted from old `base_sampler.py`:
  - `linear_energy_interpolation()` — `(1-t)*E_source + t*E_target`
  - `linear_energy_interpolation_gradients()` — autograd for x_grad and t_grad
  - `mcmc_kernel()` — dispatches to ULA or MALA kernel from `src.models.samplers.mcmc`
  - `langevin_eps_fn()` — warmup-based step size schedule
  - `update_step_size()` — adaptive step size (acceptance rate targeting)
  - `_resample()` — multinomial or systematic resampling
  - `_smc_loop()` — main loop with batched MCMC, ESS monitoring, DDP-aware resampling
- DDP support: particles sharded across ranks, `all_gather` for ESS and resampling
- All plotting methods from old `SMCSampler` preserved (energy plots, weights, eps, acceptance rate, particle survival)
- Returns `{"proposal": ..., "resampled": ..., "smc": ...}`

### Naming
User requested `BaseSampler` / `SNISSampler` / `SMCSampler` instead of the "Strategy" suffix from REFACTOR.md.

### Note on naming conflict
The old `SMCSampler` class lives in `src/models/samplers/base_sampler.py`. The new `SMCSampler` lives in `src/models/samplers/smc_sampler.py`. These coexist during the transition. The old one will be removed in Phase 8.

### Verified
- All three classes import successfully
- No modifications to existing code — existing tests unaffected

## Phase 6: Create evaluator and callback — DONE

Created new files alongside existing code (no modifications to existing files):

### 6a. `src/evaluation/evaluator.py`
- `Evaluator` class — extracted from `TransferableBoltzmannGeneratorLitModule.evaluate()` and `BaseDataModule.metrics_and_plots()`
- Constructor params: `fix_symmetry`, `drop_unfixable_symmetry`, `num_eval_samples`, `do_plots`
- `evaluate(sequence, samples_data_dict, eval_context, log_image_fn, prefix)` → metrics dict
- `_fix_chirality(proposal_data, true_data, topology, prefix)` → (fixed_data, symmetry_metrics)
- Handles: chirality detection/fixing, metric computation via `evaluate_peptide_data`, all plot generation (Ramachandran, TICA, energies, atom distances, CoM norms)

### 6b. `src/callbacks/sampling_evaluation.py`
- `SamplingEvaluationCallback(Callback)` — replaces `evaluate_all`, `on_eval_epoch_end` from Lightning module
- Constructor: takes `evaluator: Evaluator`
- `on_validation_epoch_end` / `on_test_epoch_end` → delegates to `_evaluate_epoch`
- `_evaluate_epoch` handles EMA weight swap (backup/copy/restore) via `isinstance(net, EMA)` check
- `_evaluate_all` loops sequences, calls `pl_module.sampler.sample()`, then `evaluator.evaluate()`
- DDP: broadcasts aggregated metrics from rank 0 via `broadcast_object_list`
- Free functions: `detach_and_cpu()`, `add_aggregate_metrics()` — extracted from Lightning module

### 6c. `configs/callbacks/sampling_evaluation.yaml`
- Hydra config for instantiating the callback with nested evaluator

### New directory
- Created `src/callbacks/` package with `__init__.py`

### Verified
- Both classes import successfully
- Hydra instantiation works
- No modifications to existing code — all existing tests unaffected

## Phase 7: Migrate Lightning modules — DONE

### 7a. `src/models/base_lightning_module.py` (new file)
- `BaseLightningModule(LightningModule)` — replaces `TransferableBoltzmannGeneratorLitModule`
- Constructor: `net`, `optimizer`, `scheduler`, `prior`, `sampler` (optional), `ema_decay`, `compile`, etc.
- Abstract methods: `model_step()`, `sample_proposal()`, `proposal_energy()`
- Training lifecycle: `training_step`, `validation_step`, `test_step`, `eval_step`
- EMA support: `optimizer_step` updates EMA, `state_dict` excludes teacher
- Self-improve: `on_train_epoch_start` generates samples via sampler, fills replay buffer
- Teacher/distillation: `on_fit_start` creates teacher copy for distill loss
- Gradient monitoring: `on_after_backward`, `on_before_optimizer_step`

### 7b. Migrated `NormalizingFlowLitModule`
- Now inherits from `BaseLightningModule` instead of `TransferableBoltzmannGeneratorLitModule`
- `sample_proposal()` — inlined `_generate_samples_internal`, includes invertibility logging
- `proposal_energy()` — returns raw energy (COM adjustment moved to samplers)
- Removed unused `math`, `scipy` imports

### 7c. Migrated `FlowMatchLitModule`
- Now inherits from `BaseLightningModule`
- Implements `model_step()`, `sample_proposal()`, `proposal_energy()`

### 7d. Updated model configs
- `configs/model/default.yaml` — added `sampler: null`
- `configs/model/normalizing_flow.yaml`, `flow_matching.yaml`, `flow_matching++.yaml` — verified correct `_target_` paths

### 7e. Updated callback configs
- `configs/callbacks/default.yaml` — includes `sampling_evaluation`
- `configs/callbacks/eval.yaml` — includes `sampling_evaluation`

### 7f. Updated all experiment configs
- All training configs: sampler set via `model.sampler` with `_target_` pointing to new sampler classes
- All evaluation configs: sampler configs migrated from old `sampling_config` dict to `model.sampler` object
- `configs/debug/limit.yaml` — replaced old `model.sampling_config` with `callbacks.sampling_evaluation.evaluator.num_eval_samples`

### 7g. Updated tests
- `test_snis_mwe.py` — replaces SMC sampler with SNIS for ULA/MALA configs (replicates old `smc_sampler.enabled = False`)
- `test_smc_mwe.py`, `test_self_improve_mwe.py` — config overrides updated to `cfg.model.sampler.*`

### 7h. Fixed import paths
- `base_sampling_strategy` → `base_sampler_class` in `base_lightning_module.py`, `snis_sampler.py`, `smc_sampler.py`

### 7i. Moved COM adjustment to samplers
- Removed COM adjustment from `NormalizingFlowLitModule.proposal_energy()`
- `SMCSampler` wraps `source_energy_fn` with COM adjustment when `use_com_adjustment=True`
- `SNISSampler` already handled COM adjustment on `log_q`

### 7j. Extracted normalization to standalone functions
- Created `src/data/normalization.py` with `normalize(x, std)` and `unnormalize(x, std)`
- Updated `Evaluator.evaluate()` to take `normalization_std: torch.Tensor` instead of `unnormalize` callable
- Updated samplers to use `unnormalize(samples, std)` with `std = model.trainer.datamodule.std`
- Updated `single_peptide_datamodule.py` and `transferable_peptide_datamodule.py` to call standalone functions
- Removed `normalize`/`unnormalize` methods from `BaseDataModule`

### 7k. Simplified callback
- Merged `_evaluate_epoch` and `_evaluate_all` into single `evaluate()` method on `SamplingEvaluationCallback`

## Phase 8: Cleanup — DONE

### 8a. Deleted old monolith
- Deleted `src/models/transferable_boltzmann_generator_module.py` (752 lines)

### 8b. Deleted old sampler
- Deleted `src/models/samplers/base_sampler.py` (old `SMCSampler` class, only imported by deleted monolith)

### 8c. Removed dead code from BaseDataModule
- Removed `metrics_and_plots()` method and its 7 unused imports (plot functions, evaluate_peptide_data, TicaModel, SamplesData)

### 8d. Cleaned up stale comments
- Removed references to `TransferableBoltzmannGeneratorLitModule` from `sampling_evaluation.py` and `evaluator.py`

### Verified
All 7 tests pass:
- `test_train_mwe[cpu-ecnf++_AAA]`
- `test_train_mwe[cpu-tarflow_AAA]`
- `test_snis_mwe[cpu-tarflow_AAA_ula]`
- `test_snis_mwe[cpu-prose_up_to_8aa_snis]`
- `test_smc_mwe[cpu-tarflow_AAA_ula]`
- `test_smc_mwe[cpu-prose_up_to_8aa_mala]`
- `test_self_improve_mwe[cpu-prose_up_to_8aa_self_improve]`

## Phase 9: Sampler Ownership — DONE

Moved samplers from model to callbacks.

- Removed `sampler` constructor parameter and `self.sampler` from `BaseLightningModule`
- `SamplingEvaluationCallback` owns its sampler instance
- Model's `_populate_buffer` uses the model's own sampler field (set only for self-improve configs)
- Experiment configs specify samplers on callbacks, not model

## Phase 10: Training Script + Init/Resume Utils — DONE

### 10a. Created `src/utils/init_resume_utils.py`

New utility module with:

| Function | Purpose |
|---|---|
| `load_state_dict_from_ckpt(path)` | Extract `state_dict` from a `.ckpt` Lightning checkpoint |
| `load_state_dict_from_file(path)` | Load raw state dict from a `.pth` file |
| `load_state_dict_from_hf(hf_filepath, scratch_dir)` | Download from HuggingFace Hub, return state dict |
| `resolve_init(init_ckpt_path, init_hf_state_dict_path, scratch_dir)` | Resolve init weights from checkpoint or HF (mutually exclusive), returns `Optional[dict]` |
| `resolve_init_or_resume(resume_ckpt_path, init_ckpt_path, init_hf_state_dict_path, scratch_dir)` | Preemptible semantics: if resume checkpoint exists and is loadable, resume; otherwise delegate to `resolve_init`. Returns `(fit_ckpt_path, init_state_dict)` |
| `augment_state_dict_for_teacher(sd)` | Copy `net.*` keys to `teacher.*` for distillation init from a student-only state dict |

### 10b. Updated `src/train.py`

- Config params renamed: `ckpt_path` → `resume_ckpt_path`, `init_state_dict_hf_path` → split into `init_hf_state_dict_path` + `init_ckpt_path`
- Uses `resolve_init_or_resume()` → gets `(fit_ckpt_path, init_state_dict)`
- If `init_state_dict` is returned, augments for teacher if model has one, then loads into model
- Passes `fit_ckpt_path` to `trainer.fit()` for full Lightning resume
- Preemptible: if `resume_ckpt_path` exists on disk and is loadable, ignores init weights and resumes

### 10c. Updated `src/eval.py`

- Uses `resolve_init()` with `cfg.get("ckpt_path")` and `cfg.get("hf_state_dict_path")`
- XOR enforced by `resolve_init` assertion
- Weights loaded into model before `trainer.validate()`/`trainer.test()`
- Removed `os` import, replaced `download_weights` with `resolve_init`

### 10d. Config changes

**`configs/train.yaml`:**
```yaml
init_hf_state_dict_path: null
init_ckpt_path: null
resume_ckpt_path: null
```

**`configs/eval.yaml`:**
```yaml
ckpt_path: null
hf_state_dict_path: null
```

### 10e. Renamed eval experiment config keys

All 14 eval experiment configs: `state_dict_hf_path` → `hf_state_dict_path`

### 10f. Updated self-improve config

`configs/experiment/evaluation/transferable/prose_up_to_8aa_self_improve.yaml`: `init_state_dict_hf_path` → `init_hf_state_dict_path` (routes through `train.py`)

### 10g. Updated `src/models/base_lightning_module.py`

- Teacher created in `__init__` (not `on_fit_start`), so it's part of the model state from construction
- Removed `state_dict()` override — teacher weights now persist in checkpoints (needed for `augment_state_dict_for_teacher` on init)
- `_build_ema_net_copy` → `_build_net_copy(use_ema_if_available)` with proper fallback
- `configure_optimizers` filters to `requires_grad=True` only (excludes frozen teacher)
- `_populate_buffer` runs every epoch in `on_train_epoch_start` (not `on_fit_start`)
- EMA assertions moved from `__init__` to `on_train_epoch_start` (trainer not available at construction)
- `torch.no_grad()` → `torch.inference_mode()` for teacher forward pass

### 10h. Bug fixes during implementation

- Fixed `logger` vs `logging` naming mismatch in `init_resume_utils.py`
- Fixed `logging.info` → `logger.info` in `train.py`
- Fixed `_build_net_copy` returning `None` when `use_ema_if_available=True` but no EMA callback found
- Fixed `_has_ema_callback` accessing `self.trainer` property (raises `RuntimeError`) during `__init__` — moved assertion to `on_train_epoch_start`

### Verified

All 6 integration tests pass:

| Test | Status |
|---|---|
| `test_snis_mwe[cpu-tarflow_AAA_ula]` | PASS |
| `test_snis_mwe[cpu-prose_up_to_8aa_snis]` | PASS |
| `test_smc_mwe` (2 params) | PASS |
| `test_self_improve_mwe[cpu-prose_up_to_8aa_self_improve]` | PASS |
| `test_train_mwe[cpu-ecnf++_AAA]` | PASS |
| `test_train_mwe[cpu-tarflow_AAA]` | PASS |

---

## All Phases Complete

The refactor is finished. All original design goals from REFACTOR.md have been achieved.
