# Refactoring Plan

## Problem Statement

The base Lightning module `TransferableBoltzmannGeneratorLitModule` (752 lines) is a monolith that mixes:
- Training logic (optimizer, scheduler, EMA, gradient checks)
- Evaluation orchestration (`evaluate_all`, `evaluate`, `metrics_and_plots`)
- Sampling orchestration (`batched_generate_samples`, `generate_and_resample`)
- DDP coordination (`self.all_gather` in `generate_samples`)
- Chirality fixing (symmetry checks inlined in `evaluate`)
- Importance resampling + CoM adjustment (inlined in `evaluate`)
- Self-improvement loop (`on_train_epoch_start` with buffer management)

Additionally:
- SNIS vs SMC is controlled by if/else branches and config flags, not composable objects
- The prior (`NormalDistribution`) is hardcoded in the base constructor
- `BaseDataModule.metrics_and_plots()` couples data management with evaluation
- `SinglePeptideDataModule` wraps single values in dicts "for compatibility with transferable case"
- SMC is not DDP-compatible (`@pytest.mark.skipif(torch.cuda.device_count() > 1)`)

## Design Principles

1. **Minimal public surface on Lightning modules.** The model exposes only what Lightning requires plus two abstract methods for samplers.
2. **Distributed-agnostic modules.** The model generates samples on the local rank. DDP gathering is the sampler's responsibility (the sampler receives the full Lightning module to access `all_gather`, `trainer.world_size`, etc.).
3. **Distributed-friendly SMC.** Particles are sharded across ranks; MCMC runs locally; resampling coordinates globally.
4. **No illusion of generality in datamodules.** Both datamodule implementations provide the same interface honestly, without "unused compatibility argument" patterns.

## Target Lightning Module API

### `BaseLightningModule(LightningModule)` — replaces `TransferableBoltzmannGeneratorLitModule`

```python
class BaseLightningModule(LightningModule):
    def __init__(self, net, optimizer, scheduler, prior, sampler, ema_decay, compile, ...):
        # prior: Prior — injected, not hardcoded
        # sampler: BaseSamplingStrategy — injected strategy object

    # ── Lightning required (not abstract, implemented in base) ──────────
    def training_step(self, batch, batch_idx) -> loss
    def validation_step(self, batch, batch_idx)
    def test_step(self, batch, batch_idx)
    def configure_optimizers(self)
    def setup(self, stage)

    # ── Abstract: must be implemented by FlowMatch / NormalizingFlow ────
    def model_step(self, batch) -> loss
        """Compute training loss. Called by training_step."""

    def sample_proposal(self, num_samples, system_cond) -> tuple[Tensor, Tensor]:
        """Generate num_samples on the LOCAL rank only. No all_gather.
        Returns (samples, log_q) where log_q is the raw proposal log-likelihood.
        system_cond is a dict with optional 'permutations' and 'encodings'."""

    def proposal_energy(self, samples, system_cond) -> Tensor:
        """Compute -log q(x) for given samples. Must be consistent with
        sample_proposal's log_q (i.e. proposal_energy ≈ -log_q)."""
```

**What moves out of the Lightning module:**
- `evaluate_all`, `evaluate` → `SamplingEvaluationCallback`
- `batched_generate_samples` → `BaseSamplingStrategy.sample_proposal_in_batches()`
- `generate_and_resample` → `SNISStrategy` (used by self-improvement)
- `add_aggregate_metrics`, `detach_and_cpu` → free functions in callback module
- `log_image` → passed as callable where needed, not a model method
- `on_validation_epoch_end` evaluation trigger → `SamplingEvaluationCallback.on_validation_epoch_end`
- All chirality/symmetry logic → `Evaluator`
- CoM energy adjustment → `resampling_utils.com_energy_adjustment()` (free function)

**What stays:**
- `training_step` / `validation_step` / `test_step` — compute loss only, no eval
- `configure_optimizers` — optimizer + scheduler setup
- `on_train_epoch_start` — self-improvement sample generation (calls `self.sampler`)
- `on_after_backward` / `on_before_optimizer_step` — gradient logging/nan checks
- `optimizer_step` — EMA update
- `on_fit_start` — teacher network copy for distillation
- `state_dict` — teacher exclusion override

### Child implementations

**`NormalizingFlowLitModule(BaseLightningModule)`:**
- `model_step(batch)` — MLE loss + optional distillation + optional energy KL
- `sample_proposal(num_samples, system_cond)` — sample prior, reverse through NF, return `(x, log_q)`. NO `all_gather`.
- `proposal_energy(samples, system_cond)` — forward through NF, compute `prior_energy - fwd_logdets`

**`FlowMatchLitModule(BaseLightningModule)`:**
- `model_step(batch)` — flow matching MSE loss
- `sample_proposal(num_samples, system_cond)` — sample prior, integrate ODE, return `(x, log_q)`. NO `all_gather`.
- `proposal_energy(samples, system_cond)` — reverse integrate, compute energy

## Target Sampling Strategy API

```python
class BaseSamplingStrategy:
    def sample(self, model, system_cond, target_energy_fn, prefix) -> dict[str, SamplesData]:
        """Run the full sampling pipeline. Handles DDP gathering internally.
        Returns dict mapping strategy names to SamplesData, e.g.:
          {"proposal": ..., "resampled": ..., "smc": ...}
        The model is the full LightningModule — used for all_gather, device, world_size."""

    def sample_proposal_in_batches(self, model, num_samples, batch_size, system_cond):
        """Helper: calls model.sample_proposal in batches, then all_gathers."""
```

**`ProposalOnlyStrategy(BaseSamplingStrategy)`:**
- Calls `sample_proposal_in_batches`, returns `{"proposal": SamplesData}`
- Used during training eval when no resampling is desired

**`SNISStrategy(BaseSamplingStrategy)`:**
- Generates proposal, gathers across DDP
- Computes target energy, proposal log_q
- Applies CoM adjustment if configured
- Clips logits if configured
- Resamples via importance weights
- Returns `{"proposal": SamplesData, "resampled": SamplesData}`
- Config: `use_com_adjustment`, `logit_clip_filter`, `proposal_batch_size`, `num_samples`

**`SMCStrategy(BaseSamplingStrategy)`:**
- Generates proposal via SNIS first
- Runs SMC loop with configurable MCMC kernel
- **DDP-aware**: each rank holds `N/world_size` particles; MCMC runs locally; `all_gather` weights for ESS; `all_gather` + scatter for resampling
- Config: `kernel_type` ("ula"/"mala"/"hmc"), `langevin_eps`, `num_timesteps`, `ess_threshold`, `warmup`, `batch_size` (for gradient batching within a rank)
- Returns `{"proposal": ..., "resampled": ..., "smc": ...}`

## Target Datamodule API

### `BaseDataModule`

Remove `metrics_and_plots()`. Keep: `normalize()`, `unnormalize()`, dataloaders, `prepare_eval()`.

### `prepare_eval` unified signature

Both datamodules return the same namedtuple/dataclass:

```python
@dataclass
class EvalContext:
    true_samples: Tensor          # normalized
    system_cond: dict              # {"permutations": ..., "encodings": ...} or both None
    target_energy_fn: Callable     # maps normalized samples -> energy (scalar per sample)
    tica_model: Optional[TicaModel]
    topology: mdtraj.Topology
```

```python
# SinglePeptideDataModule
def prepare_eval(self, sequence: str, stage: str) -> EvalContext:
    ...  # sequence is always self.hparams.sequence; no "unused" disclaimer

# TransferablePeptideDataModule
def prepare_eval(self, sequence: str, stage: str) -> EvalContext:
    ...  # loads NPZ for the given sequence
```

**SinglePeptideDataModule cleanup:**
- Remove `# For compatibility with transferable case` comments
- `topology_dict` → just `self.topology` (the dict wrapping moves to where it's needed, or is removed)
- `val_sequences` / `test_sequences` are legitimate; they're always `[self.hparams.sequence]`
- Use `PeptidesDataset` instead of `TensorDataset` (or keep a minimal inline Dataset class — `TensorDataset` is 20 lines)

## Target Evaluator

```python
class Evaluator:
    def __init__(self, fix_symmetry, drop_unfixable_symmetry, num_eval_samples):
        ...

    def evaluate(self, sequence, samples_data_dict, eval_context, log_image_fn=None):
        """
        Args:
            sequence: peptide sequence string
            samples_data_dict: dict[str, SamplesData] from sampler
            eval_context: EvalContext from datamodule.prepare_eval()
            log_image_fn: optional image logging callable
        Returns:
            (metrics_dict, plots_dict)
        """
        # Fix chirality
        # Compute metrics via evaluate_peptide_data
        # Generate plots if log_image_fn provided
```

## Target Callback

```python
class SamplingEvaluationCallback(Callback):
    def __init__(self, evaluator: Evaluator):
        ...

    def on_validation_epoch_end(self, trainer, pl_module):
        for sequence in trainer.datamodule.val_sequences:
            eval_context = trainer.datamodule.prepare_eval(sequence, "val")
            samples_dict = pl_module.sampler.sample(
                pl_module, eval_context.system_cond,
                eval_context.target_energy_fn, prefix=f"val/{sequence}"
            )
            metrics, plots = self.evaluator.evaluate(
                sequence, samples_dict, eval_context, log_image_fn=...
            )
            for k, v in metrics.items():
                trainer.log(k, v, ...)
        # aggregate metrics across sequences
```

---

## Execution Phases

### Phase 1: Extract utility functions (zero risk)

These are new files with no behavior change; existing code is updated to import from the new location.

**1a. `src/utils/chirality.py`** (new file)

Move from `src/models/utils.py`:
- `create_adjacency_list(distance_matrix, atom_types) -> list`
- `get_atom_types(topology) -> Tensor`
- `get_adj_list(topology) -> Tensor`
- `find_chirality_centers(adj_list, atom_types, num_h_atoms=2) -> Tensor`
- `compute_chirality_sign(coords, chirality_centers) -> Tensor`
- `check_symmetry_change(true_coords, pred_coords, adj_list, atom_types) -> Tensor`
- `get_symmetry_change(true_samples, pred_samples, topology) -> Tensor`

Update `src/models/utils.py` to re-export: `from src.utils.chirality import get_symmetry_change, ...`
Update `src/models/transferable_boltzmann_generator_module.py` line 22: import from `src.utils.chirality`.

**1b. `src/utils/resampling_utils.py`** (new file)

Move from `src/models/utils.py`:
- `resample(samples, logits, return_index=False)` — the existing function (lines 7-18)

New standalone functions extracted from `NormalizingFlowLitModule.com_energy_adjustment` (normalizing_flow_module.py:89-100):
```python
def com_energy_adjustment(x: Tensor, com_std: float) -> Tensor:
    """Compute center-of-mass energy correction.
    x: (batch, num_atoms, 3), com_std: scalar std of CoM norms."""
    com = x.mean(dim=1)
    com_norm = com.norm(dim=-1)
    return com_norm**2 / (2 * com_std**2) - torch.log(
        com_norm**2 / (math.sqrt(2) * com_std**3 * scipy.special.gamma(3/2))
    )
```

Update `src/models/utils.py` to re-export `resample`.

**1c. `src/utils/dataclasses.py`** (new file)

Copy content from `src/utils/data_types.py` (the `SamplesData` dataclass, unchanged).
Add:
```python
@dataclass
class EvalContext:
    true_samples: torch.Tensor
    system_cond: dict
    target_energy_fn: Callable
    tica_model: object  # Optional[TicaModel]
    topology: object    # mdtraj.Topology
```

Update `src/utils/data_types.py` to become: `from src.utils.dataclasses import *`
Update imports in `src/data/base_datamodule.py` (line 15) and `src/models/transferable_boltzmann_generator_module.py` (line 22).

**Verify after Phase 1:**
```
PYTEST_TRAINER=cpu uv run pytest -vv -s tests/integrations/test_train_mwe.py -k "tarflow_AAA"
PYTEST_TRAINER=cpu uv run pytest -vv -s tests/integrations/test_snis_mwe.py -k "tarflow_AAA_ and snis"
```

---

### Phase 2: Extract priors to package (low risk)

**2a. Create `src/models/priors/` package**

`src/models/priors/prior.py`:
```python
from abc import ABC, abstractmethod
import torch

class Prior(ABC):
    @abstractmethod
    def sample(self, num_samples: int, num_atoms: int, mask=None, device="cpu") -> torch.Tensor:
        ...
    @abstractmethod
    def energy(self, x: torch.Tensor, mask=None) -> torch.Tensor:
        ...
```

`src/models/priors/normal_distribution.py`:
Copy the full `NormalDistribution` class from current `src/models/priors.py` (lines 1-43, without the `__main__` test block). Add `Prior` as base class:
```python
from src.models.priors.prior import Prior
class NormalDistribution(Prior):
    ...  # existing implementation unchanged
```

`src/models/priors/__init__.py`:
```python
from src.models.priors.prior import Prior
from src.models.priors.normal_distribution import NormalDistribution
```

**2b. Replace the file with the package**

Must be done atomically via git:
```bash
git rm src/models/priors.py
mkdir -p src/models/priors
# create the 3 files above
git add src/models/priors/
```

The import in `transferable_boltzmann_generator_module.py` line 19 (`from src.models.priors import NormalDistribution`) continues to work because the `__init__.py` re-exports it.

**Verify after Phase 2:**
```
PYTEST_TRAINER=cpu uv run pytest -vv -s tests/integrations/test_train_mwe.py -k "ecnf++_AAA"
PYTEST_TRAINER=cpu uv run pytest -vv -s tests/integrations/test_train_mwe.py -k "tarflow_AAA"
```

---

### Phase 3: Extract MCMC kernels (low risk)

**3a. Create `src/models/samplers/mcmc/` package**

`src/models/samplers/mcmc/__init__.py`:
```python
from src.models.samplers.mcmc.ula import ula_kernel
from src.models.samplers.mcmc.mala import mala_kernel
from src.models.samplers.mcmc.hmc import hmc_kernel
```

`src/models/samplers/mcmc/ula.py`:
Extract from `SMCSamplerULA` (ula_sampler.py). The kernel becomes a free function:
```python
def ula_kernel(energy_interpolation_fn, energy_interpolation_grad_fn, t, x, logw, dt, eps):
    """ULA MCMC kernel step.
    Args:
        energy_interpolation_fn: (t, x) -> energy
        energy_interpolation_grad_fn: (t, x) -> (x_grad, t_grad)  # ULA needs t_grad
        t: current SMC time (scalar)
        x: particles (batch, atoms, 3)
        logw: log importance weights (batch,)
        dt: time step size
        eps: Langevin step size
    Returns: (x, logw, acceptance_rate)
    """
    energy_grad_x, energy_grad_t = energy_interpolation_grad_fn(t, x)
    dx = -eps * energy_grad_x + math.sqrt(2 * eps) * torch.randn_like(x)
    dlogw = -energy_grad_t * dt
    x = x + dx
    logw = logw + dlogw
    return x, logw, torch.ones(1).mean()
```

`src/models/samplers/mcmc/mala.py`:
Extract from `SMCSamplerMALA` (mala_sampler.py). Same pattern:
```python
def mala_kernel(energy_interpolation_fn, energy_interpolation_grad_fn, t, x, logw, dt, eps):
    """MALA MCMC kernel step.
    energy_interpolation_grad_fn here returns x_grad only (no t_grad).
    """
    energy_fn = energy_interpolation_fn
    grad_fn = energy_interpolation_grad_fn

    energy_grad_x = grad_fn(t, x)
    dx = -eps * energy_grad_x + math.sqrt(2 * eps) * torch.randn_like(x)
    x_proposal = x + dx
    s = torch.max(torch.zeros_like(t), t - dt)
    dlogw = -energy_fn(t, x) + energy_fn(s, x)

    # Metropolis-Hastings
    energy_grad_x_proposal = grad_fn(t, x_proposal)
    E_proposal = energy_fn(t, x_proposal)
    E = energy_fn(t, x)
    logp = -E_proposal + E
    logp += -0.5 * torch.sum(((x - x_proposal + eps * energy_grad_x_proposal)**2).reshape(x.shape[0], -1), dim=-1) / (2 * eps)
    logp -= -0.5 * torch.sum(((x_proposal - x + eps * energy_grad_x)**2).reshape(x.shape[0], -1), dim=-1) / (2 * eps)

    u = torch.rand_like(logp)
    mask = (logp > torch.log(u))[..., None, None].float()
    x = mask * x_proposal + (1 - mask) * x
    logw = logw + dlogw
    acceptance_rate = mask.mean()
    return x, logw, acceptance_rate
```

`src/models/samplers/mcmc/hmc.py`:
Extract from `SMCSamplerHMC` (hmc_sampler.py):
```python
def leapfrog(grad_energy_fn, t, x, v, dt):
    v = v - 0.5 * dt * grad_energy_fn(t, x)
    x = x + dt * v
    v = v - 0.5 * dt * grad_energy_fn(t, x)
    return x, v

def hmc_kernel(energy_interpolation_fn, energy_interpolation_grad_fn, t, x, logw, dt, eps):
    """HMC MCMC kernel step."""
    energy_fn = energy_interpolation_fn
    grad_fn = energy_interpolation_grad_fn
    norm = lambda _v: torch.sum(_v**2, dim=-1)

    v = torch.randn_like(x)
    s = torch.max(torch.zeros_like(t), t - dt)
    dlogw = -energy_fn(t, x) + energy_fn(s, x)

    x_proposal, v_proposal = leapfrog(grad_fn, t, x, v, eps)

    logp = -0.5 * norm(v_proposal) + 0.5 * norm(v) - energy_fn(t, x_proposal) + energy_fn(t, x)
    u = torch.rand_like(logp)
    mask = (logp > torch.log(u))[..., None].float()
    x = mask * x_proposal + (1 - mask) * x
    logw = logw + dlogw
    acceptance_rate = mask.mean()
    return x, logw, acceptance_rate
```

**3b. Wire old samplers to use new kernels**

Update `SMCSamplerULA.mcmc_kernel` to delegate:
```python
from src.models.samplers.mcmc.ula import ula_kernel
class SMCSamplerULA(SMCSampler):
    def mcmc_kernel(self, source_energy, target_energy, t, x, logw, dt):
        eps = self.langevin_eps_fn(t)
        energy_fn = lambda _t, _x: self.linear_energy_interpolation(source_energy, target_energy, _t, _x)
        grad_fn = lambda _t, _x: self._ula_grad(source_energy, target_energy, _t, _x)
        return ula_kernel(energy_fn, grad_fn, t, x, logw, dt, eps)
```

Same pattern for MALA and HMC. This validates the extraction is behaviorally identical.

**Verify after Phase 3:**
```
PYTEST_TRAINER=cpu uv run pytest -vv -s tests/integrations/test_smc_mwe.py -k "tarflow_AAA_"
```

---

### Phase 4: Unify datamodule `prepare_eval` (low risk)

**4a. Update `SinglePeptideDataModule.prepare_eval` to return `EvalContext`**

Current signature: `prepare_eval(self, sequence, prefix)` returns 5-tuple.
New: returns `EvalContext`. Change at `single_peptide_datamodule.py:194-233`:

```python
from src.utils.dataclasses import EvalContext

def prepare_eval(self, sequence: str, stage: str) -> EvalContext:
    if stage == "test":
        true_samples = self.data_test.data
    elif stage == "val":
        true_samples = self.data_val.data
    else:
        raise ValueError(f"Unknown stage: {stage}")

    tica_model = get_tica_model(true_samples, self.topology)
    true_samples = true_samples[:: len(true_samples) // self.hparams.num_eval_samples]
    true_samples = self.normalize(true_samples)

    potential = self.setup_potential()
    energy_fn = lambda x: potential.energy(self.unnormalize(x)).flatten()

    return EvalContext(
        true_samples=true_samples,
        system_cond={"permutations": None, "encodings": None},
        target_energy_fn=energy_fn,
        tica_model=tica_model,
        topology=self.topology,
    )
```

**4b. Update `TransferablePeptideDataModule.prepare_eval` to return `EvalContext`**

Change at `transferable_peptide_datamodule.py:259-298`:

```python
from src.utils.dataclasses import EvalContext

def prepare_eval(self, sequence: str, stage: str) -> EvalContext:
    data_path = self.val_data_path if sequence in self.val_sequences else self.test_data_path
    npz = np.load(os.path.join(data_path, f"{len(sequence)}AA", f"{sequence}_subsampled.npz"))

    true_samples = torch.from_numpy(npz["positions"])
    tica_model = TicaModel(projection=npz["tica_projection"], mean=npz["tica_mean"], dim=npz["tica_dim"])
    true_samples = self.normalize(true_samples)

    potential = self.setup_potential(sequence)
    energy_fn = lambda x: potential.energy(self.unnormalize(x)).flatten()

    return EvalContext(
        true_samples=true_samples,
        system_cond={"permutations": self.permutations_dict[sequence], "encodings": self.encodings_dict[sequence]},
        target_energy_fn=energy_fn,
        tica_model=tica_model,
        topology=self.topology_dict[sequence],
    )
```

**4c. Update callers of `prepare_eval`**

In `transferable_boltzmann_generator_module.py`, `evaluate_all` (line 310) currently unpacks a 5-tuple:
```python
true_samples, permutations, encodings, energy_fn, tica_model = self.datamodule.prepare_eval(...)
```
Update to:
```python
eval_ctx = self.datamodule.prepare_eval(prefix=prefix, sequence=sequence)
# Then pass eval_ctx fields to self.evaluate(...)
```

Similarly update `generate_and_resample` (line 593) and `NormalizingFlowLitModule.model_step` (line 61).

**Note**: The old call sites pass `prefix="val"` or `prefix="test"`. The new param is `stage`, same values. Both datamodules previously had the "other" arg as unused — now both use both args.

**Verify after Phase 4:**
```
PYTEST_TRAINER=cpu uv run pytest -vv -s tests/integrations/test_snis_mwe.py -k "tarflow_AAA_ and snis"
PYTEST_TRAINER=cpu uv run pytest -vv -s tests/integrations/test_smc_mwe.py -k "tarflow_AAA_"
PYTEST_TRAINER=cpu uv run pytest -vv -s tests/integrations/test_self_improve_mwe.py
PYTEST_TRAINER=cpu uv run pytest -vv -s tests/integrations/test_train_mwe.py -k "ecnf++_AAA"
PYTEST_TRAINER=cpu uv run pytest -vv -s tests/integrations/test_train_mwe.py -k "tarflow_AAA"
```
(All 5 tests — this touches the interface used everywhere.)

---

### Phase 5: Create sampling strategies (medium risk)

**5a. `src/models/samplers/base_sampling_strategy.py`**

```python
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import torch
from tqdm import tqdm
from src.utils.dataclasses import SamplesData

if TYPE_CHECKING:
    from src.models.base_lightning_module import BaseLightningModule

class BaseSamplingStrategy:
    def __init__(self, num_samples: int, proposal_batch_size: int):
        self.num_samples = num_samples
        self.proposal_batch_size = proposal_batch_size

    @abstractmethod
    def sample(self, model, system_cond, target_energy_fn, prefix="") -> dict[str, SamplesData]:
        ...

    def sample_proposal_in_batches(self, model, num_samples, system_cond):
        """Generate num_samples total across all DDP ranks, in batches.
        Handles all_gather internally. Returns gathered (samples, log_q)."""
        world_size = model.trainer.world_size if model.trainer else 1
        local_total = num_samples // world_size

        all_samples, all_log_q = [], []
        remaining = local_total
        while remaining > 0:
            batch_n = min(self.proposal_batch_size, remaining)
            samples, log_q = model.sample_proposal(batch_n, system_cond)
            all_samples.append(samples)
            all_log_q.append(log_q)
            remaining -= batch_n

        local_samples = torch.cat(all_samples, dim=0)
        local_log_q = torch.cat(all_log_q, dim=0)

        # Gather across DDP ranks
        if world_size > 1:
            local_samples = model.all_gather(local_samples).reshape(-1, *local_samples.shape[1:])
            local_log_q = model.all_gather(local_log_q).reshape(-1, *local_log_q.shape[1:])

        return local_samples, local_log_q
```

**5b. `src/models/samplers/snis_strategy.py`**

```python
class SNISStrategy(BaseSamplingStrategy):
    def __init__(
        self,
        num_samples: int,
        proposal_batch_size: int,
        use_com_adjustment: bool = False,
        logit_clip_filter: float = None,
    ):
        super().__init__(num_samples, proposal_batch_size)
        self.use_com_adjustment = use_com_adjustment
        self.logit_clip_filter = logit_clip_filter

    def sample(self, model, system_cond, target_energy_fn, prefix="") -> dict[str, SamplesData]:
        samples, log_q = self.sample_proposal_in_batches(model, self.num_samples, system_cond)
        target_energy = target_energy_fn(samples)

        # CoM adjustment
        if self.use_com_adjustment:
            com_std = samples.mean(dim=1).std()
            log_q = log_q + com_energy_adjustment(samples, com_std)

        # Importance weights
        logits = -target_energy - log_q

        # Clip logits
        if self.logit_clip_filter:
            keep = logits <= torch.quantile(logits, 1 - self.logit_clip_filter)
            samples, target_energy, logits, log_q = samples[keep], target_energy[keep], logits[keep], log_q[keep]

        # Resample
        _, resampling_index = resample(samples, logits, return_index=True)

        unnorm = model.trainer.datamodule.unnormalize
        proposal_data = SamplesData(unnorm(samples), target_energy)
        resampled_data = SamplesData(unnorm(samples[resampling_index]), target_energy[resampling_index], logits=logits)

        return {"proposal": proposal_data, "resampled": resampled_data}
```

**5c. `src/models/samplers/smc_strategy.py`**

```python
class SMCStrategy(BaseSamplingStrategy):
    def __init__(
        self,
        num_samples: int,
        proposal_batch_size: int,
        kernel_type: str,           # "ula", "mala", "hmc"
        langevin_eps: float,
        num_timesteps: int,
        ess_threshold: float,
        systematic_resampling: bool = False,
        warmup: float = 0.0,
        gradient_batch_size: int = 128,
        input_energy_filter_cutoff: float = None,
        # SNIS params for initial proposal
        use_com_adjustment: bool = False,
        logit_clip_filter: float = None,
    ):
        ...

    def sample(self, model, system_cond, target_energy_fn, prefix=""):
        # 1. Generate proposal via SNIS
        snis = SNISStrategy(self.num_samples, self.proposal_batch_size,
                           self.use_com_adjustment, self.logit_clip_filter)
        snis_result = snis.sample(model, system_cond, target_energy_fn, prefix)

        # 2. Build source energy = model.proposal_energy (conditioned on system_cond)
        source_energy_fn = lambda x: model.proposal_energy(x, system_cond)

        # 3. Run SMC loop on proposal samples
        #    DDP: each rank holds N/world_size particles
        proposal_samples = ...  # normalized samples from snis
        smc_samples, smc_logits = self._smc_loop(
            proposal_samples, source_energy_fn, target_energy_fn, model
        )

        unnorm = model.trainer.datamodule.unnormalize
        smc_data = SamplesData(unnorm(smc_samples), target_energy_fn(smc_samples), logits=smc_logits)
        return {**snis_result, "smc": smc_data}

    def _smc_loop(self, proposal_samples, source_energy, target_energy, model):
        """Run SMC. In DDP mode, particles are sharded across ranks."""
        world_size = model.trainer.world_size if model.trainer else 1

        # Shard particles across ranks
        if world_size > 1:
            rank = model.local_rank
            chunk_size = len(proposal_samples) // world_size
            X = proposal_samples[rank * chunk_size : (rank + 1) * chunk_size]
        else:
            X = proposal_samples

        A = torch.zeros(X.shape[0], device=X.device)
        timesteps = torch.linspace(0, 1, self.num_timesteps + 1)
        kernel_fn = {"ula": ula_kernel, "mala": mala_kernel, "hmc": hmc_kernel}[self.kernel_type]

        t_previous = 0.0
        for j, t in enumerate(timesteps[:-1]):
            dt = t - t_previous
            # Run kernel in gradient-computation batches
            X, A = self._batched_kernel_step(kernel_fn, source_energy, target_energy, t, X, A, dt)

            # Check ESS — requires global weights
            if world_size > 1:
                global_A = model.all_gather(A).reshape(-1)
            else:
                global_A = A
            ess = sampling_efficiency(global_A)

            if ess < self.ess_threshold and j + 1 != self.num_timesteps:
                # Global resampling: gather all particles, resample, scatter
                if world_size > 1:
                    global_X = model.all_gather(X).reshape(-1, *X.shape[1:])
                    global_X, _ = self._resample(global_X, global_A)
                    # Scatter back to ranks
                    X = global_X[rank * chunk_size : (rank + 1) * chunk_size]
                else:
                    X, _ = self._resample(X, A)
                A = torch.zeros(X.shape[0], device=X.device)

            t_previous = t

        # Final gather + resample
        if world_size > 1:
            X = model.all_gather(X).reshape(-1, *X.shape[1:])
            A = model.all_gather(A).reshape(-1)
        X, _ = self._resample(X, A)
        return X, A
```

**Note on DDP SMC**: The key insight is that MCMC kernels are embarrassingly parallel across particles. Only resampling requires global coordination. The `all_gather` at resampling steps is the communication bottleneck, but resampling happens infrequently (only when ESS drops below threshold).

---

### Phase 6: Create evaluator and callback (medium risk)

**6a. `src/evaluation/evaluator.py`**

```python
class Evaluator:
    def __init__(self, fix_symmetry=True, drop_unfixable_symmetry=False, num_eval_samples=10_000):
        ...

    def evaluate(self, sequence, samples_data_dict, eval_context, log_image_fn=None):
        """
        samples_data_dict: {"proposal": SamplesData, "resampled": SamplesData, "smc"?: SamplesData}
        eval_context: EvalContext from datamodule
        Returns: (metrics_dict, plots_dict)
        """
        metrics = {}
        true_data = SamplesData(eval_context.true_samples_unnormalized, ...)

        for name, data in samples_data_dict.items():
            if data is None:
                continue
            # Fix chirality
            if self.fix_symmetry:
                data = self._fix_chirality(data, true_data, eval_context.topology)
            # Compute metrics
            metrics.update(evaluate_peptide_data(true_data, data, ...))
            # Plots
            if log_image_fn:
                plot_ramachandran(log_image_fn, ...)
                plot_tica(log_image_fn, ...)

        return metrics, {}
```

The chirality-fixing logic is extracted verbatim from `transferable_boltzmann_generator_module.py` lines 441-474.

**6b. `src/callbacks/sampling_evaluation.py`**

As shown in the Target Callback section above. This takes over the role of `evaluate_all` and `on_eval_epoch_end` from the base Lightning module.

**6c. `configs/callbacks/sampling_evaluation.yaml`**

```yaml
sampling_evaluation:
  _target_: src.callbacks.sampling_evaluation.SamplingEvaluationCallback
  evaluator:
    _target_: src.evaluation.evaluator.Evaluator
    fix_symmetry: true
    drop_unfixable_symmetry: false
    num_eval_samples: 10_000
```

**Verify after Phase 6:**
Wire into one test config manually and run. Then revert config for the next phase.

---

### Phase 7: Migrate Lightning modules (high risk — most careful)

This is the critical phase. Each sub-step should be a separate commit.

**7a. Create `src/models/base_lightning_module.py`**

New file implementing `BaseLightningModule` as described in the Target API section. Copy the following from `transferable_boltzmann_generator_module.py`:
- `__init__` (modified: accept `prior` and `sampler` args instead of constructing prior internally)
- `training_step` (unchanged)
- `eval_step`, `validation_step`, `test_step` (unchanged)
- `configure_optimizers` (unchanged)
- `setup` (unchanged)
- `on_after_backward`, `on_before_optimizer_step`, `optimizer_step` (unchanged)
- `on_fit_start` (self-improvement teacher copy — unchanged)
- `on_train_epoch_start` (self-improvement — modified to use `self.sampler`)
- `state_dict` override (unchanged)

**Remove** from the new base:
- `evaluate_all`, `evaluate` — handled by callback
- `on_validation_epoch_end`, `on_test_epoch_end` — these no longer trigger eval (callback does)
- `batched_generate_samples`, `generate_samples` — replaced by `sample_proposal`
- `generate_and_resample` — replaced by sampler
- `predict_step` — remove or simplify
- `add_aggregate_metrics`, `detach_and_cpu`, `log_image` — moved to callback/utils

**Key constructor change:**
```python
# Old:
def __init__(self, net, optimizer, scheduler, datamodule, smc_sampler, sampling_config, ...):
    self.prior = NormalDistribution(datamodule.hparams.num_dimensions, mean_free=self.hparams.mean_free_prior)
    self.smc_sampler = smc_sampler

# New:
def __init__(self, net, optimizer, scheduler, prior, sampler, ema_decay, compile, ...):
    self.prior = prior    # injected
    self.sampler = sampler  # injected (can be None for training-only)
```

**Note**: The `datamodule` arg is **removed** from the constructor. Currently it's used for:
- `self.datamodule.hparams.num_dimensions` → now provided by `prior` (prior already has `num_dimensions`)
- `self.datamodule.unnormalize(x)` → accessed via `self.trainer.datamodule.unnormalize(x)` when needed
- `self.datamodule.prepare_eval(...)` → accessed via `self.trainer.datamodule` in callback
- `self.datamodule.topology_dict[seq]` → accessed via `eval_context.topology`

**7b. Migrate `NormalizingFlowLitModule`**

Change parent class. Implement the two abstract methods:

```python
class NormalizingFlowLitModule(BaseLightningModule):
    def sample_proposal(self, num_samples, system_cond):
        """Generate num_samples on local rank. No all_gather."""
        encodings = system_cond.get("encodings")
        permutations = system_cond.get("permutations")
        num_atoms = encodings["atom_type"].size(0) if encodings else self.prior.num_dimensions  # need num_atoms

        data_dim = num_atoms * 3  # num_dimensions=3

        prior_samples = self.prior.sample(num_samples, num_atoms, device=self.device)
        prior_log_q = -self.prior.energy(prior_samples) * data_dim

        # Broadcast permutations/encodings to batch
        _permutations = ...  # same logic as current generate_samples
        _encodings = ...

        with torch.no_grad():
            x_pred = self.net.reverse(prior_samples, _permutations, encodings=_encodings)
            _, fwd_logdets = self.net(x_pred, _permutations, encodings=_encodings)
            fwd_logdets = fwd_logdets * data_dim

        log_q = prior_log_q.flatten() + fwd_logdets.flatten()
        return x_pred, log_q

    def proposal_energy(self, samples, system_cond):
        """Compute proposal energy. Same as current proposal_energy but takes system_cond."""
        encodings = system_cond.get("encodings")
        permutations = system_cond.get("permutations")
        # ... same implementation as current, without com_energy_adjustment
        # (CoM adjustment moves to sampler)
```

**Important**: `com_energy_adjustment` is **removed** from this class. It becomes a free function in `resampling_utils` and is called by `SNISStrategy`.

Update `configs/model/normalizing_flow.yaml`:
```yaml
defaults:
  - default
  - net: tarflow
  - scheduler: cosine

_target_: src.models.normalizing_flow_module.NormalizingFlowLitModule

prior:
  _target_: src.models.priors.normal_distribution.NormalDistribution
  num_dimensions: ${data.num_dimensions}
  mean_free: false

optimizer:
  lr: 1e-4
  betas: [0.9, 0.95]
  weight_decay: 4e-4
ema_decay: 0.999
```

**7c. Migrate `FlowMatchLitModule`**

Same pattern. `sample_proposal` wraps the ODE integration. `proposal_energy` wraps reverse flow.

Update `configs/model/flow_matching.yaml` and `flow_matching++.yaml` similarly (with `mean_free: true`).

**7d. Update model default config**

`configs/model/default.yaml`:
```yaml
defaults:
  - optimizer: adam_w

sampler: null   # was smc_sampler: null + sampling_config: {...}

compile: false
```

The `sampling_config` dict is **eliminated**. Its fields move to the sampler config:
- `num_proposal_samples`, `num_test_proposal_samples` → `sampler.num_samples`
- `batch_size` → `sampler.proposal_batch_size`
- `use_com_adjustment` → `sampler.use_com_adjustment`
- `clip_reweighting_logits` → `sampler.logit_clip_filter`
- `num_smc_samples` → `sampler.num_samples` (on SMCStrategy)

**7e. Update all experiment configs**

Each eval experiment config changes from:
```yaml
model:
  sampling_config:
    use_com_adjustment: True
    batch_size: 5_000
    num_test_proposal_samples: 1_000_000
    clip_reweighting_logits: 0.002
  smc_sampler: null
```
To:
```yaml
model:
  sampler:
    _target_: src.models.samplers.snis_strategy.SNISStrategy
    num_samples: 1_000_000
    proposal_batch_size: 5_000
    use_com_adjustment: true
    logit_clip_filter: 0.002
```

And SMC configs change from:
```yaml
model:
  smc_sampler:
    _target_: src.models.samplers.ula_sampler.SMCSamplerULA
    enabled: True
    ...
```
To:
```yaml
model:
  sampler:
    _target_: src.models.samplers.smc_strategy.SMCStrategy
    kernel_type: ula
    ...
```

**7f. Update entry points (`eval.py`, `self_improve.py`)**

In `eval.py` line 81: currently passes `datamodule=datamodule` to model constructor.
Remove the `datamodule=` arg since the new base doesn't accept it.
The model accesses datamodule via `self.trainer.datamodule` after `trainer.fit/validate/test` is called.

Same for `self_improve.py` line 70.

In `train.py`: same change.

**7g. Update test configs**

The test fixtures override config keys like:
- `cfg.model.sampling_config.num_test_proposal_samples = 25` → `cfg.model.sampler.num_samples = 25`
- `cfg.model.smc_sampler.num_timesteps = 10` → `cfg.model.sampler.num_timesteps = 10`
- `cfg.model.smc_sampler.enabled = False` → no longer needed (sampler is null or SNIS)

**Verify after Phase 7 (run ALL tests):**
```
PYTEST_TRAINER=cpu uv run pytest -vv -s tests/integrations/test_snis_mwe.py -k "tarflow_AAA_ and snis"
PYTEST_TRAINER=cpu uv run pytest -vv -s tests/integrations/test_smc_mwe.py -k "tarflow_AAA_"
PYTEST_TRAINER=cpu uv run pytest -vv -s tests/integrations/test_self_improve_mwe.py
PYTEST_TRAINER=cpu uv run pytest -vv -s tests/integrations/test_train_mwe.py -k "ecnf++_AAA"
PYTEST_TRAINER=cpu uv run pytest -vv -s tests/integrations/test_train_mwe.py -k "tarflow_AAA"
```

---

### Phase 8: Cleanup (low risk)

Delete old files:
- `src/models/transferable_boltzmann_generator_module.py`
- `src/models/samplers/base_sampler.py`
- `src/models/samplers/hmc_sampler.py`
- `src/models/samplers/mala_sampler.py`
- `src/models/samplers/ula_sampler.py`
- `src/utils/data_types.py` (shim)
- `src/data/datasets/tensor_dataset.py` (if not already removed)

Remove re-export shims from `src/models/utils.py`. Delete functions that have been extracted. If anything remains, keep it; otherwise delete the file.

Remove `metrics_and_plots` from `BaseDataModule`. Remove its evaluation-related imports (lines 8-14 of base_datamodule.py).

Remove the SMC DDP skip marker from `test_smc_mwe.py`:
```python
# Remove this line:
@pytest.mark.skipif(torch.cuda.device_count() > 1, reason="Not yet implemented for DDP")
```

Delete `refactor/` directory.

**Verify after Phase 8:** All 5 test commands pass.

---

## Phase 9 — Sampler Ownership: Move Samplers from Model to Callbacks

**Design principle:** Sampling is an external procedure performed on the model. The model exposes the sampling interface; callbacks own and configure the sampler instances.

### 9a. Remove sampler from BaseLightningModule

**Files:** `src/models/base_lightning_module.py`, `configs/model/default.yaml`

- Remove `sampler` constructor parameter and `self.sampler` attribute
- Remove `on_train_epoch_start` self-improve logic (moves to callback in 9c)
- Model keeps only: `model_step`, `sample_proposal`, `proposal_energy`, training lifecycle, EMA, teacher/distill (used in `model_step`)
- Remove `sampler: null` from `configs/model/default.yaml`

### 9b. SamplingEvaluationCallback owns its sampler

**Files:** `src/callbacks/sampling_evaluation.py`, `configs/callbacks/sampling_evaluation.yaml`

- Add `sampler` field to `SamplingEvaluationCallback.__init__`
- Call `self.sampler.sample(pl_module, ...)` instead of `pl_module.sampler.sample(pl_module, ...)`
- Update callback config:

```yaml
# configs/callbacks/sampling_evaluation.yaml
sampling_evaluation:
  _target_: src.callbacks.sampling_evaluation.SamplingEvaluationCallback
  sampler: null  # overridden by experiment configs
  evaluator:
    _target_: src.evaluation.evaluator.Evaluator
    ...
```

### 9c. New SamplingSelfImprovementCallback

**Files:** `src/callbacks/self_improve.py` (new), `configs/callbacks/self_improve.yaml` (new)

Callback responsibilities:
- Owns its own sampler instance (can differ from eval sampler)
- `on_train_epoch_start`: generate samples, fill replay buffer
- Handles EMA weight swap for sampling (same pattern as eval callback)
- Can call `pl_module.ensure_teacher_initialized()` if needed

```yaml
# configs/callbacks/self_improve.yaml
self_improve:
  _target_: src.callbacks.self_improve.SamplingSelfImprovementCallback
  sampler: null  # overridden by experiment configs
  num_proposal_samples: 200_000
```

Key implementation notes:
- Avoid sampler field mutation for temporary overrides (e.g. don't temporarily change `sampler.num_samples`). Pass overrides as args to `sample()` instead.
- Teacher creation and distill loss stay on the LightningModule since they're used in `model_step`.
- Self-improve is enabled/disabled by adding/removing the callback.

### 9d. Update experiment configs

**Files:** All experiment configs under `configs/experiment/`

Move sampler configs from `model.sampler` to callback configs. Two options:

**Option 1 (recommended):** Callback config owns sampler config directly.

```yaml
callbacks:
  sampling_evaluation:
    sampler:
      _target_: src.models.samplers.snis_sampler.SNISSampler
      num_samples: 50_000
      proposal_batch_size: 2_500
```

**Option 2:** Keep sampler specs under `model.*` and reference them via Hydra interpolation.

```yaml
model:
  eval_sampler:
    _target_: src.models.samplers.snis_sampler.SNISSampler
    num_samples: 50_000

callbacks:
  sampling_evaluation:
    sampler: ${model.eval_sampler}
```

### 9e. Update tests

**Files:** `tests/integrations/test_snis_mwe.py`, `tests/integrations/test_smc_mwe.py`, `tests/integrations/test_self_improve_mwe.py`

- Update config overrides from `cfg.model.sampler.*` to the new callback config paths
- Test self-improve by verifying the callback is present and functional

### Invariants

- Sampler API unchanged: `sampler.sample(model, ...)` where model implements `sample_proposal()` and `proposal_energy()`
- No `pl_module.sampler` attribute anywhere
- Evaluation and self-improve sampling are procedures implemented as callbacks; can be enabled/disabled by adding/removing callbacks
- Distributed behavior unchanged: sampling runs inside Lightning callback hooks where DDP is already initialized

---

## Phase Dependency Graph

```
Phase 1 (utilities)    ─┐
Phase 2 (priors)       ─┤─ independent, can run in any order
Phase 3 (mcmc kernels) ─┘
         │
Phase 4 (datamodule prepare_eval) ── depends on Phase 1c (EvalContext dataclass)
         │
Phase 5 (sampling strategies) ── depends on Phases 1b, 1c, 3
         │
Phase 6 (evaluator + callback) ── depends on Phases 1a, 1c
         │
Phase 7 (migrate lightning modules) ── depends on Phases 2, 4, 5, 6
         │
Phase 8 (cleanup) ── depends on Phase 7
         │
Phase 9 (sampler ownership → callbacks) ── depends on Phase 7
```

## Test Command → Phase Mapping

| Test Command | What it exercises | Required after phases |
|---|---|---|
| `test_train_mwe -k "ecnf++_AAA"` | FlowMatch training | 1, 2, 4, 7c |
| `test_train_mwe -k "tarflow_AAA"` | NormalizingFlow training | 1, 2, 4, 7b |
| `test_snis_mwe -k "tarflow_AAA_ and snis"` | SNIS eval pipeline | 1, 2, 4, 5b, 6, 7b, 7e |
| `test_smc_mwe -k "tarflow_AAA_"` | SMC eval pipeline | 1, 2, 3, 4, 5c, 6, 7b, 7e |
| `test_self_improve_mwe` | Self-improvement loop | 1, 2, 4, 5b, 7b, 7e |

## Known Risks

1. **Hydra config `_target_` paths**: When class paths change, all referencing YAML files must update. Missing one causes silent instantiation failures.
2. **Circular imports**: `BaseSamplingStrategy` needs type hints for `BaseLightningModule`. Use `TYPE_CHECKING` guard.
3. **`self.hparams` access patterns**: Currently child modules access `self.hparams.sampling_config.*`. After refactor, these fields move to `self.sampler.*`. Any missed reference will be a runtime error.
4. **DDP SMC**: The distributed resampling logic is new code (not extracted from existing). Needs careful testing beyond the CPU-only tests we have.
5. **`datamodule` removal from constructor**: Currently the model accesses `self.datamodule.*` extensively. All such accesses must be replaced with `self.trainer.datamodule.*` (available after `trainer.fit/validate/test` but NOT during `__init__`). Code that runs in `__init__` and needs datamodule info (like `num_dimensions`) must get it from the injected `prior` or other constructor args.
