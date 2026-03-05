from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch

from transferable_samplers.models.base_lightning_module import BaseLightningModule
from transferable_samplers.models.priors.prior import Prior
from transferable_samplers.utils.dataclasses import SourceEnergyConfig, SystemCond
from transferable_samplers.utils.pylogger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=False)


class NormalizingFlowModule(BaseLightningModule):
    """Normalizing flow generative model trained via maximum likelihood.

    Maps target conformations to a simple prior via an invertible network,
    trained to maximize log-likelihood. Generation uses the reverse pass of
    the invertible network.

    Supports optional teacher regularization for fine-tuning: a frozen copy
    of the initial network penalizes the model's log-density from drifting
    too far from the teacher's predictions.

    See ``BaseLightningModule`` for inherited args.

    Args:
        teacher_regularize_weight: Weight for teacher regularization loss.
            If > 0, a frozen copy of the initial network is used as teacher.
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        prior: Prior,
        scheduler: Any = None,
        compile_net: bool = False,
        source_energy_config: SourceEnergyConfig | None = None,
        train_from_buffer: bool = False,
        mean_free_prior: bool = False,
        teacher_regularize_weight: float = 0.0,
    ) -> None:
        assert not mean_free_prior, "Mean free prior is not supported for normalizing flows"

        super().__init__(
            net=net,
            optimizer=optimizer,
            scheduler=scheduler,
            prior=prior,
            compile_net=compile_net,
            source_energy_config=source_energy_config,
            train_from_buffer=train_from_buffer,
            mean_free_prior=mean_free_prior,
        )

        self.teacher_regularize_weight = teacher_regularize_weight

        if self.teacher_regularize_weight > 0:
            logger.info(f"Using teacher regularization with weight {self.teacher_regularize_weight:.3f}")
            self.teacher = deepcopy(self.net)
            for param in self.teacher.parameters():
                param.requires_grad_(False)
            self.teacher.eval()

    def compute_primary_loss(self, batch: dict[str, Any]) -> torch.Tensor:
        """Compute per-sample negative log-likelihood loss."""
        x = batch["x"]
        assert len(x.shape) == 3, "molecules must be a pointcloud (batch_size, num_atoms, 3)"

        encodings = batch.get("encodings")
        permutations = batch.get("permutations")
        mask = batch.get("mask")

        z_pred, dlogp = self.net(x, permutations=permutations, encodings=encodings, mask=mask)
        logp_pred_z = self.prior.logp(z_pred, mask=mask)
        logq = logp_pred_z + dlogp
        return -logq

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Compute training loss with optional teacher regularization."""
        if self.train_from_buffer:
            assert self._buffer is not None, "Buffer must be set for training from buffer"
            batch = self._buffer.sample(batch["x"].shape[0], device=self.device)

        per_sample_loss = self.compute_primary_loss(batch)
        loss = self.normalize_by_system_dim(per_sample_loss, batch["x"], batch.get("mask"))

        self.log("train/mle_loss", loss.item(), prog_bar=True, sync_dist=True)

        if self.teacher_regularize_weight > 0:
            x = batch["x"]
            encodings = batch.get("encodings")
            permutations = batch.get("permutations")
            mask = batch.get("mask")

            with torch.inference_mode():
                z_pred_teacher, dlogp_teacher = self.teacher(
                    x, permutations=permutations, encodings=encodings, mask=mask
                )
                logq_teacher = self.prior.logp(z_pred_teacher, mask=mask) + dlogp_teacher

            logq = -per_sample_loss  # per_sample_loss is -logq
            teacher_reg = (logq - logq_teacher).pow(2).mean()
            loss = loss + self.teacher_regularize_weight * teacher_reg
            self.log("finetune/teacher_regularization", teacher_reg.item(), prog_bar=True, sync_dist=True)

        batch_value = self.train_metrics(loss)
        self.log_dict(batch_value, prog_bar=True)
        return loss

    def generate_proposal(
        self,
        net: torch.nn.Module,
        num_samples: int,
        system_cond: SystemCond | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate samples using the reverse pass of the invertible network."""
        encodings = system_cond.encodings if system_cond else None

        if encodings is None:
            # pyrefly: ignore [missing-attribute]
            num_atoms = self.trainer.datamodule.num_atoms
        else:
            num_atoms = encodings["atom_type"].size(0)

        z = self.prior.sample(num_samples, num_atoms, device=self.device)
        logp_z = self.prior.logp(z)

        batched_cond = system_cond.for_batch(num_samples, self.device) if system_cond else None
        _encodings = batched_cond.encodings if batched_cond else None
        _permutations = batched_cond.permutations if batched_cond else None

        # pyrefly: ignore [not-callable]
        x_pred, dlogp_rev = net.reverse(z, _permutations, encodings=_encodings)

        # dlogp_rev is log|det(dx/dz)| = -log|det(dz/dx)|, so logq = logp_z - dlogp_rev
        logq = logp_z - dlogp_rev

        return x_pred, -logq

    def proposal_energy(
        self,
        net: torch.nn.Module,
        x: torch.Tensor,
        system_cond: SystemCond | None = None,
    ) -> torch.Tensor:
        """Compute proposal energy (-log q) via the forward pass."""
        batched_cond = system_cond.for_batch(x.shape[0], x.device) if system_cond else None
        _encodings = batched_cond.encodings if batched_cond else None
        _permutations = batched_cond.permutations if batched_cond else None

        z_pred, dlogp = net(x, _permutations, encodings=_encodings)

        logp_z_pred = self.prior.logp(z_pred)

        logq = logp_z_pred + dlogp
        return -logq  # energy is negative log probability

    def on_train_epoch_start(self) -> None:
        """Reset metrics and validate teacher regularization config."""
        super().on_train_epoch_start()
        if self.teacher_regularize_weight > 0:
            assert not self._has_ema_callback(), (
                "EMAWeightAveraging callback should not be used with teacher regularization."
            )

    # =====================================================================
    # Additional methods (not in base class)
    # =====================================================================

    def run_model_diagnostics(
        self,
        prefix: str,
        system_cond: SystemCond | None = None,
        num_samples_invert: int = 256,
        num_samples_dlogp: int = 16,
    ) -> dict[str, float]:
        """Sample from prior and return invertibility and dlogp diagnostics.

        Invertibility: checks that reverse(z) followed by forward gives back z.
        dlogp: checks the network's dlogp against the true Jacobian dlogp (small batch).

        Args:
            prefix: Metric key prefix.
            system_cond: Optional conditioning (permutations, encodings) for transferable models.
            num_samples_invert: Number of samples for invertibility check.
            num_samples_dlogp: Number of samples for dlogp check.
        """
        encodings = system_cond.encodings if system_cond else None

        if encodings is None:
            # pyrefly: ignore [missing-attribute]
            num_atoms = self.trainer.datamodule.num_atoms
        else:
            num_atoms = encodings["atom_type"].size(0)

        # pyrefly: ignore [missing-attribute]
        data_dim = num_atoms * self.trainer.datamodule.num_dimensions
        z = self.prior.sample(num_samples_invert, num_atoms, device=self.device)

        batched_cond = system_cond.for_batch(num_samples_invert, self.device) if system_cond else None
        _encodings = batched_cond.encodings if batched_cond else None
        _permutations = batched_cond.permutations if batched_cond else None

        with torch.no_grad():  # diagnostics only, no gradient needed
            # pyrefly: ignore [not-callable]
            x_pred, _ = self.net.reverse(z, _permutations, encodings=_encodings)
            z_recon, dlogp = self.net(x_pred, _permutations, encodings=_encodings)

        diff = (z - z_recon).abs()
        metrics = {
            f"{prefix}/diagnostic/invert/mse": torch.mean((z - z_recon) ** 2).item(),
            f"{prefix}/diagnostic/invert/max_abs": torch.max(diff).item(),
            f"{prefix}/diagnostic/invert/mean_abs": torch.mean(diff).item(),
            f"{prefix}/diagnostic/invert/median_abs": torch.median(diff).item(),
        }
        for cutoff in (0.01, 0.001):
            metrics[f"{prefix}/diagnostic/invert/fail_count_{cutoff}"] = torch.sum(diff > cutoff).float().item()
            metrics[f"{prefix}/diagnostic/invert/fail_count_sample_{cutoff}"] = (
                (torch.sum(diff > cutoff, dim=1) > 0).sum().float().item()
            )

        dlogp_batch = x_pred[:num_samples_dlogp].detach()

        # Build single-sample permutations/encodings for Jacobian computation
        _perm_single = None
        if _permutations is not None:
            _perm_single = {k: v[:1] for k, v in _permutations.items()}
        _enc_single = None
        if _encodings is not None:
            _enc_single = {k: v[:1] for k, v in _encodings.items()}

        fwd_func = lambda x: self.net.forward(x, _perm_single, encodings=_enc_single)[0]
        dlogp_diffs = []
        for i in range(len(dlogp_batch)):
            x = dlogp_batch[i].unsqueeze(0).float().requires_grad_(True)
            with torch.enable_grad():
                fwd_jac = torch.autograd.functional.jacobian(fwd_func, x, vectorize=True)
                dlogp_true = torch.logdet(fwd_jac.view(data_dim, data_dim))
            dlogp_diffs.append(abs(dlogp[i] - dlogp_true).item())
        metrics[f"{prefix}/diagnostic/dlogp/mean_diff"] = sum(dlogp_diffs) / len(dlogp_diffs)
        metrics[f"{prefix}/diagnostic/dlogp/max_diff"] = max(dlogp_diffs)

        return metrics
