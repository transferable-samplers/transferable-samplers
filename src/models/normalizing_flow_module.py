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
        if self._buffer is not None:
            batch = self._buffer.sample(batch["x"].shape[0])

        assert len(batch["x"].shape) == 3, "molecules must be a pointcloud (batch_size, num_atoms, 3)"

        x1 = batch["x"]
        encodings = batch.get("encodings")
        permutations = batch.get("permutations")
        mask = batch.get("mask")

        x0, dlogp = self.net(x1, permutations=permutations, encodings=encodings, mask=mask)

        loss = self.prior.energy(x0, mask=mask).mean() - dlogp.mean()

        self.log("train/mle_loss", loss.item(), prog_bar=True, sync_dist=True)

        if self.hparams.use_distill_loss:
            with torch.inference_mode():
                x0_teacher, dlogp_teacher = self.teacher(x1, permutations=permutations, encodings=encodings, mask=mask)

            prior_log_p = -self.prior.energy(x0_teacher, mask=mask)
            logq_teacher = prior_log_p + dlogp_teacher

            logq = -self.prior.energy(x0, mask=mask) + dlogp
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

        data_dim = x.shape[1] * self.trainer.datamodule.hparams.num_dimensions

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

        # TODO need to figure out x_pred / recon names - maybe use z going forwards
        x_pred, fwd_logdets = net(x, _permutations, encodings=_encodings)

        fwd_logdets = fwd_logdets.view(-1) * data_dim  # rescale from mean to sum
        prior_energy = self.prior.energy(x_pred).view(-1) * data_dim  # rescale from mean to sum

        energy = prior_energy - fwd_logdets

        return energy

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

        data_dim = num_atoms * self.trainer.datamodule.hparams.num_dimensions

        prior_samples = self.prior.sample(num_samples, num_atoms, device=self.device)

        # need to rescale to the "sum" of the log p (the prior returns the position-wise mean)
        prior_log_q = -self.prior.energy(prior_samples) * data_dim

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
            x_pred = net.reverse(prior_samples, _permutations, encodings=_encodings)
            x_recon, fwd_logdets = net(x_pred, _permutations, encodings=_encodings)
            fwd_logdets = fwd_logdets * data_dim  # rescale from mean to sum

        log_q = prior_log_q.flatten() + fwd_logdets.flatten()

        return x_pred, log_q

    def run_model_diagnostics(
        self,
        prefix: str,
        system_cond: Optional["SystemCond"] = None,
        num_samples_invert: int = 256,
        num_samples_logdet: int = 16,
    ) -> dict:
        """Sample from prior and return invertibility and logdet diagnostics.

        Invertibility: checks that reverse(z) followed by forward gives back z.
        Logdet: checks the network's logdet against the true Jacobian logdet (small batch).

        Args:
            prefix: Metric key prefix.
            system_cond: Optional conditioning (permutations, encodings) for transferable models.
            num_samples_invert: Number of samples for invertibility check.
            num_samples_logdet: Number of samples for logdet check.
        """
        permutations = system_cond.permutations if system_cond else None
        encodings = system_cond.encodings if system_cond else None

        if encodings is None:
            num_atoms = self.trainer.datamodule.hparams.num_atoms
        else:
            num_atoms = encodings["atom_type"].size(0)

        data_dim = num_atoms * self.trainer.datamodule.hparams.num_dimensions
        prior_samples = self.prior.sample(num_samples_invert, num_atoms, device=self.device)

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
            x_pred = self.net.reverse(prior_samples, _permutations, encodings=_encodings)
            x_recon, fwd_logdets = self.net(x_pred, _permutations, encodings=_encodings)
            fwd_logdets = fwd_logdets * data_dim  # rescale from mean to sum

        # ── Invertibility metrics ─────────────────────────────────────────
        diff = (prior_samples - x_recon).abs()
        metrics = {
            f"{prefix}/diagnostic/invert/mse": torch.mean((prior_samples - x_recon) ** 2).item(),
            f"{prefix}/diagnostic/invert/max_abs": torch.max(diff).item(),
            f"{prefix}/diagnostic/invert/mean_abs": torch.mean(diff).item(),
            f"{prefix}/diagnostic/invert/median_abs": torch.median(diff).item(),
        }
        for cutoff in (0.01, 0.001):
            metrics[f"{prefix}/diagnostic/invert/fail_count_{cutoff}"] = torch.sum(diff > cutoff).float().item()
            metrics[f"{prefix}/diagnostic/invert/fail_count_sample_{cutoff}"] = (
                (torch.sum(diff > cutoff, dim=1) > 0).sum().float().item()
            )

        # ── Logdet check (small batch — Jacobian is expensive per sample) ───
        logdet_batch = x_pred[:num_samples_logdet].detach()

        # Build single-sample permutations/encodings for Jacobian computation
        _perm_single = None
        if _permutations is not None:
            _perm_single = {k: v[:1] for k, v in _permutations.items()}
        _enc_single = None
        if _encodings is not None:
            _enc_single = {k: v[:1] for k, v in _encodings.items()}

        fwd_func = lambda x: self.net.forward(x, _perm_single, encodings=_enc_single)[0]
        logdet_diffs = []
        for i in range(len(logdet_batch)):
            x = logdet_batch[i].unsqueeze(0).float().requires_grad_(True)
            with torch.enable_grad():
                fwd_jac = torch.autograd.functional.jacobian(fwd_func, x, vectorize=True)
                logdet_true = torch.logdet(fwd_jac.view(data_dim, data_dim))
            logdet_diffs.append(abs(fwd_logdets[i] - logdet_true).item())
        metrics[f"{prefix}/diagnostic/logdet/mean_diff"] = sum(logdet_diffs) / len(logdet_diffs)
        metrics[f"{prefix}/diagnostic/logdet/max_diff"] = max(logdet_diffs)

        return metrics
