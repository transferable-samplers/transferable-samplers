from typing import Optional

import torch

from src.models.base_lightning_module import BaseLightningModule
from src.utils.dataclasses import SystemCond


class NormalizingFlowLitModule(BaseLightningModule):
    def __init__(
        self,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert not self.hparams.mean_free_prior, "Mean free prior is not supported for normalizing flows"

        self.eval_ctx = None

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        if self.train_from_buffer:
            assert self._buffer is not None, "Buffer must be set for training from buffer"
            batch = self._buffer.sample(batch["x"].shape[0])

        assert len(batch["x"].shape) == 3, "molecules must be a pointcloud (batch_size, num_atoms, 3)"

        x1 = batch["x"]
        encodings = batch.get("encodings")
        permutations = batch.get("permutations")
        mask = batch.get("mask")

        if mask is not None:
            data_dim = mask.sum(dim=-1) * x1.shape[-1]
        else:
            data_dim = x1.shape[-2] * x1.shape[-1]

        z_pred, dlogp = self.net(x1, permutations=permutations, encodings=encodings, mask=mask)
        logp_pred_z = self.prior.logp(z_pred, mask=mask)
        logq = logp_pred_z + dlogp
        loss = -(logq / data_dim).mean()

        self.log("train/mle_loss", loss.item(), prog_bar=True, sync_dist=True)

        if self.hparams.use_distill_loss:
            with torch.inference_mode():
                z_pred_teacher, dlogp_teacher = self.teacher(x1, permutations=permutations, encodings=encodings, mask=mask)

            logp_z_pred_teacher = self.prior.logp(z_pred_teacher, mask=mask)
            logq_teacher = logp_z_pred_teacher + dlogp_teacher

            distill_loss = (logq - logq_teacher).pow(2).mean()
            loss = loss + self.hparams.distill_weight * distill_loss
            self.log("finetune/distill_loss", distill_loss.item(), prog_bar=True, sync_dist=True)

        batch_value = self.train_metrics(loss)
        self.log_dict(batch_value, prog_bar=True)
        return loss

    def proposal_energy(
        self, net: torch.nn.Module, x: torch.Tensor, system_cond: Optional[SystemCond] = None,
    ) -> torch.Tensor:
        permutations = system_cond.permutations if system_cond else None
        encodings = system_cond.encodings if system_cond else None

        if encodings is not None:
            _encodings = {}
            for k, v in encodings.items():
                # ensure encodings is broadcasted to batch if we pass
                # in a single peptide
                if v.ndim == 1 and x.ndim > 2:
                    v = v[None, ...].repeat(x.shape[0], *([1] * v.ndim))

                _encodings[k] = v.to(x.device)
        else:
            _encodings = None

        if permutations is not None:
            _permutations = {
                subkey: tensor.unsqueeze(0).repeat(x.shape[0], 1).to(self.device)
                for subkey, tensor in permutations.items()
            }
        else:
            _permutations = None

        z_pred, dlogp = net(x, _permutations, encodings=_encodings)

        logp_z_pred = self.prior.logp(z_pred)

        logq = logp_z_pred + dlogp
        return -logq  # energy is negative log probability

    def sample_proposal(
        self, net: torch.nn.Module, num_samples: int,
        system_cond: Optional[SystemCond] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        permutations = system_cond.permutations if system_cond else None
        encodings = system_cond.encodings if system_cond else None

        if encodings is None:
            num_atoms = self.trainer.datamodule.hparams.num_atoms
        else:
            num_atoms = encodings["atom_type"].size(0)

        z = self.prior.sample(num_samples, num_atoms, device=self.device)
        logp_z = self.prior.logp(z)

        _permutations = None
        if permutations is not None:
            _permutations = {
                subkey: tensor.unsqueeze(0).repeat(num_samples, 1).to(self.device)
                for subkey, tensor in permutations.items()
            }

        _encodings = None
        if encodings is not None:
            _encodings = {
                key: tensor.unsqueeze(0).repeat(num_samples, 1).to(self.device)
                for key, tensor in encodings.items()
            }

        with torch.no_grad():
            x_pred, dlogp_rev = net.reverse(z, _permutations, encodings=_encodings)

        # dlogp_rev is log|det(dx/dz)| = -log|det(dz/dx)|, so logq = logp_z - dlogp_rev
        logq = logp_z - dlogp_rev

        return x_pred, -logq

    def run_model_diagnostics(
        self,
        prefix: str,
        system_cond: Optional["SystemCond"] = None,
        num_samples_invert: int = 256,
        num_samples_dlogp: int = 16,
    ) -> dict:
        """Sample from prior and return invertibility and dlogp diagnostics.

        Invertibility: checks that reverse(z) followed by forward gives back z.
        dlogp: checks the network's dlogp against the true Jacobian dlogp (small batch).

        Args:
            prefix: Metric key prefix.
            system_cond: Optional conditioning (permutations, encodings) for transferable models.
            num_samples_invert: Number of samples for invertibility check.
            num_samples_dlogp: Number of samples for dlogp check.
        """
        permutations = system_cond.permutations if system_cond else None
        encodings = system_cond.encodings if system_cond else None

        if encodings is None:
            num_atoms = self.trainer.datamodule.hparams.num_atoms
        else:
            num_atoms = encodings["atom_type"].size(0)

        data_dim = num_atoms * self.trainer.datamodule.hparams.num_dimensions
        z = self.prior.sample(num_samples_invert, num_atoms, device=self.device)

        _permutations = None
        if permutations is not None:
            _permutations = {
                subkey: tensor.unsqueeze(0).repeat(num_samples_invert, 1).to(self.device)
                for subkey, tensor in permutations.items()
            }

        _encodings = None
        if encodings is not None:
            _encodings = {
                key: tensor.unsqueeze(0).repeat(num_samples_invert, 1).to(self.device)
                for key, tensor in encodings.items()
            }

        with torch.no_grad():
            x_pred, _ = self.net.reverse(z, _permutations, encodings=_encodings)
            z_recon, dlogp = self.net(x_pred, _permutations, encodings=_encodings)

        # ── Invertibility metrics ─────────────────────────────────────────
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

        # ── dlogp check (small batch — Jacobian is expensive per sample) ───
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
