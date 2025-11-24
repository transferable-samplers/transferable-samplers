import math
from typing import Optional

import scipy
import torch

from src.models.transferable_boltzmann_generator_module import TransferableBoltzmannGeneratorLitModule


class NormalizingFlowLitModule(TransferableBoltzmannGeneratorLitModule):
    def __init__(
        self,
        energy_kl_weight: float = 0.0,
        log_invertibility_error: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """Initialize a `NormalizingFlowLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__(*args, **kwargs)
        assert not self.hparams.mean_free_prior, "Mean free prior is not supported for normalizing flows"

        self.eval_encodings = None
        self.eval_energy = None

    def model_step(self, batch: torch.Tensor, log: bool = True) -> torch.Tensor:
        x1 = batch["x"]
        encodings = batch.get("encodings")
        permutations = batch.get("permutations")
        mask = batch.get("mask")

        x0, dlogp = self.net(x1, permutations=permutations, encodings=encodings, mask=mask)

        loss = self.prior.energy(x0, mask=mask).mean() - dlogp.mean()

        if log:
            self.log("train/mle_loss", loss.item(), prog_bar=True, sync_dist=True)

        if self.hparams.use_distill_loss and self.hparams.self_improve and self.training:
            with torch.no_grad():
                x0_teacher, dlogp_teacher = self.teacher(x1, permutations=permutations, encodings=encodings, mask=mask)

            prior_log_p = -self.prior.energy(x0_teacher, mask=mask)
            logq_teacher = prior_log_p + dlogp_teacher

            logq = -self.prior.energy(x0, mask=mask) + dlogp
            distill_loss = (logq - logq_teacher).pow(2).mean()
            loss = loss + self.hparams.distill_weight * distill_loss
            if log:
                self.log("finetune/distill_loss", distill_loss.item(), prog_bar=True, sync_dist=True)

        if self.hparams.energy_kl_weight:
            assert self.hparams.eval_sequence is not None, "Eval sequence name should be set"

            if self.eval_encodings is None:
                # on first step, we need to prepare the eval encodings
                model_inputs, _, self.eval_energy = self.datamodule.prepare_eval(self.hparams.eval_sequence)
                self.eval_encodings = model_inputs.encodings

            samples, log_q_theta, _ = self.generate_samples(
                self.hparams.energy_kl_batch_size,
                encodings=self.eval_encodings,
                permutations=self.eval_permutations,
            )

            log_p = -self.eval_energy(samples)

            if log:
                self.log("train/log_p_mean", log_p.mean(), prog_bar=True, sync_dist=True)
                self.log("train/log_p_median", log_p.median(), prog_bar=True, sync_dist=True)

                self.log("train/log_q_theta_mean", log_q_theta.mean(), prog_bar=True, sync_dist=True)
                self.log("train/log_q_theta_median", log_q_theta.median(), prog_bar=True, sync_dist=True)

            num_atoms = self.eval_encodings["atom_type"].size(0)
            data_dim = num_atoms * self.datamodule.hparams.num_dimensions
            energy_loss = (log_q_theta - log_p).mean() / data_dim

            loss = loss + self.hparams.energy_kl_weight * energy_loss

            if log:
                self.log("train/energy_loss", energy_loss.item(), prog_bar=True, sync_dist=True)

        return loss

    def com_energy_adjustment(self, x: torch.Tensor) -> torch.Tensor:
        assert self.proposal_com_std is not None, "Center of mass std should be set"

        sigma = self.proposal_com_std

        com = x.mean(dim=1, keepdim=False)
        com_norm = com.norm(dim=-1)
        com_energy = com_norm**2 / (2 * sigma**2) - torch.log(
            com_norm**2 / (math.sqrt(2) * sigma**3 * scipy.special.gamma(3 / 2))
        )

        return com_energy

    def proposal_energy(self, x: torch.Tensor, permutations, encodings: dict[str, torch.Tensor]) -> torch.Tensor:
        data_dim = x.shape[1] * self.datamodule.hparams.num_dimensions  # is the product num_atoms * num_dimensions
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
        x_pred, fwd_logdets = self.net(x, _permutations, encodings=_encodings)

        fwd_logdets = fwd_logdets.view(-1) * data_dim  # rescale from mean to sum
        prior_energy = self.prior.energy(x_pred).view(-1) * data_dim  # rescale from mean to sum

        energy = prior_energy - fwd_logdets

        if self.hparams.sampling_config.use_com_adjustment:
            com_energy = self.com_energy_adjustment(x)
            energy = energy - com_energy

        return energy

    def batchify_model_input(self, model_inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            key: tensor.unsqueeze(0).repeat(local_batch_size, 1).to(self.device)
            for key, tensor in model_inputs.items()
        }
        return model_inputs

    def invertibility_metrics(self, z, z_recon):

        metrics = {}

        metrics["mse"] = torch.mean((z - z_recon) ** 2)
        metrics["max_abs"] = torch.max(abs(z - z_recon))
        metrics["mean_abs"] = torch.mean(abs(z - z_recon))
        metrics["median_abs"] = torch.median(abs(z - z_recon))

        for cutoff in [0.01, 0.001, 0.0001]:
            metrics[f"pointwise_fail_at_{cutoff}"] = torch.sum(abs(z - z_recon) > cutoff)
            metrics[f"sample_fail_at_{cutoff}"] = (torch.sum(abs(z - z_recon) > cutoff, dim=1) > 0).sum()

        return metrics

    def all_gather_batch_dim(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.all_gather(tensor).reshape(-1, *tensor.shape[1:])

    def sample_proposal(
        self,
        num_samples: int,
        model_inputs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample proposal from the model.

        :param num_samples: The number of samples to generate.
        :param model_inputs: The model inputs to use for generating samples.
        :return: A tuple containing the generated samples, the prior samples, and the log
            probability.
        """

        encodings = model_inputs.get("encodings")
        permutations = model_inputs.get("permutations")

        if encodings is None:
            num_atoms = self.datamodule.hparams.num_atoms
        else:
            num_atoms = encodings["atom_type"].size(0)

        data_dim = num_atoms * self.datamodule.hparams.num_dimensions

        local_batch_size = num_samples // self.trainer.world_size
        prior_samples = self.prior.sample(local_batch_size, num_atoms, device=self.device)

        # need to rescale to the "sum" of the log p (the prior returns the position-wise mean)
        prior_samples_energy = -self.prior.energy(prior_samples_energy) * data_dim

        _permutations = self.batchify_model_input(permutations)
        _encodings = self.batchify_model_input(encodings)

        proposal_samples = self.net.reverse(prior_samples, _permutations, encodings=_encodings)
        z_recon, logdets = self.net(proposal_samples, _permutations, encodings=_encodings)
        logdets = logdets * data_dim  # rescale from mean to sum

        invertibility_metrics = self.invertibility_metrics(z, z_recon)

        x_pred = self.all_gather_batch_dim(x_pred)
        logdets = self.all_gather_batch_dim(logdets)
        z_energy = self.all_gather_batch_dim(z_energy)
        prior_samples = self.all_gather_batch_dim(prior_samples)

        proposal_energy = - (z_energy.flatten() + logdets.flatten())

        return x_pred, proposal_energy, z

if __name__ == "__main__":
    _ = NormalizingFlowLitModule(None, None, None, None)
