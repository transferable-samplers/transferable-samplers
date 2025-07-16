import math
from typing import Optional

import scipy
import torch

from src.models.transferable_boltzmann_generator_module import TransferableBoltzmannGeneratorLitModule

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


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

    def model_step(
        self,
        batch: torch.Tensor,
    ) -> torch.Tensor:
        x1 = batch["x"]
        encodings = batch.get("encodings", None)
        permutations = batch.get("permutations", None)

        x0, dlogp = self.net(x1, permutations=permutations, encodings=encodings)

        if permutations is not None:
            mask = batch["permutations"]["atom"].get("mask", None)
        else:
            mask = None

        loss = self.prior.energy(x0, mask=mask).mean() - dlogp.mean()

        self.log("train/mle_loss", loss.item(), prog_bar=True, sync_dist=True)

        if self.hparams.energy_kl_weight:
            assert self.hparams.eval_sequence is not None, "Eval sequence name should be set"

            if self.eval_encodings is None:
                # on first step, we need to prepare the eval encodings
                _, self.eval_encodings, self.eval_energy = self.datamodule.prepare_eval(self.hparams.eval_sequence)

            samples, log_q_theta, _ = self.generate_samples(
                self.hparams.energy_kl_batch_size,
                encodings=self.eval_encodings,
                permutations=self.eval_permutations,
            )

            log_p = -self.eval_energy(samples)

            self.log("train/log_p_mean", log_p.mean(), prog_bar=True, sync_dist=True)
            self.log("train/log_p_median", log_p.median(), prog_bar=True, sync_dist=True)

            self.log("train/log_q_theta_mean", log_q_theta.mean(), prog_bar=True, sync_dist=True)
            self.log("train/log_q_theta_median", log_q_theta.median(), prog_bar=True, sync_dist=True)

            num_atoms = self.eval_encodings["atom_type"].size(0)
            data_dim = num_atoms * self.datamodule.hparams.num_dimensions
            energy_loss = (log_q_theta - log_p).mean() / data_dim

            loss = loss + self.hparams.energy_kl_weight * energy_loss
            self.log("train/energy_loss", energy_loss.item(), prog_bar=True, sync_dist=True)

        return loss

    def com_energy_adjustment(self, x: torch.Tensor) -> torch.Tensor:
        assert self.proposal_com_std is not None, "Center of mass std should be set"

        sigma = self.proposal_com_std

        com = self.datamodule.center_of_mass(x)
        com_norm = com.norm(dim=-1)
        com_energy = com_norm**2 / (2 * sigma**2) - torch.log(
            com_norm**2 / (math.sqrt(2) * sigma**3 * scipy.special.gamma(3 / 2))
        )

        return com_energy

    def proposal_energy(self, x: torch.Tensor, encodings: dict[str, torch.Tensor]) -> torch.Tensor:
        data_dim = x.shape[1]  # is the product num_atoms * num_dimensions
        if encodings is not None:
            _encodings = {}
            for k, v in encodings.items():
                # ensure encodings is broadcasted to batch if we pass
                # in a single peptide
                if v.shape[0] != x.shape[0]:
                    v = v[None, ...].repeat(x.shape[0], *([1] * v.ndim))

                _encodings[k] = v.to(x.device)
        else:
            _encodings = None
        # TODO need to figure out x_pred / recon names - maybe use z going forwards
        x_pred, fwd_logdets = self.net(x, encodings=_encodings)

        fwd_logdets = fwd_logdets.view(-1) * data_dim  # rescale from mean to sum
        prior_energy = self.prior.energy(x_pred).view(-1) * data_dim  # rescale from mean to sum

        energy = prior_energy - fwd_logdets

        if self.hparams.sampling_config.use_com_adjustment:
            com_energy = self.com_energy_adjustment(x)
            energy = energy - com_energy

        return energy

    def generate_samples(
        self,
        batch_size: int,
        permutations: dict[str, torch.Tensor],
        encodings: Optional[dict[str, torch.Tensor]] = None,
        n_timesteps: int = None,
        dummy_ll=False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate samples from the model.

        :param batch_size: The batch size to use for generating samples.
        :param n_timesteps: The number of timesteps to use when generating samples.
        :param device: The device to use for generating samples.
        :return: A tuple containing the generated samples, the prior samples, and the log
            probability.
        """

        if encodings is None:
            num_atoms = self.datamodule.hparams.num_atoms
        else:
            num_atoms = encodings["atom_type"].size(0)

        data_dim = num_atoms * self.datamodule.hparams.num_dimensions

        local_batch_size = batch_size // self.trainer.world_size
        prior_samples = self.prior.sample(local_batch_size, num_atoms, device=self.device)

        # need to rescale to the "sum" of the log p (the prior returns the position-wise mean)
        prior_log_q = -self.prior.energy(prior_samples) * data_dim

        with torch.no_grad():
            x_pred = self.net.reverse(prior_samples, permutations, encodings=encodings)
            x_recon, fwd_logdets = self.net(x_pred, permutations, encodings=encodings)
            fwd_logdets = fwd_logdets * data_dim  # rescale from mean to sum

            # TODO refector these all into a metrics
            self.log("invert/mse", torch.mean((prior_samples - x_recon) ** 2), sync_dist=True)
            self.log(
                "invert/max_abs",
                torch.max(abs(prior_samples - x_recon)),
                sync_dist=True,
            )
            self.log(
                "invert/mean_abs",
                torch.mean(abs(prior_samples - x_recon)),
                sync_dist=True,
            )
            self.log(
                "invert/median_abs",
                torch.median(abs(prior_samples - x_recon)),
                sync_dist=True,
            )
            cutoff = 0.01
            self.log(
                f"invert/fail_count_{cutoff}",
                torch.sum(abs(prior_samples - x_recon) > cutoff).sum().float(),
                sync_dist=True,
            )
            self.log(
                f"invert/fail_count_sample_{cutoff}",
                (torch.sum(abs(prior_samples - x_recon) > cutoff, dim=1) > 0).sum().float(),
                sync_dist=True,
            )
            cutoff = 0.001
            self.log(
                f"invert/fail_count_{cutoff}",
                torch.sum(abs(prior_samples - x_recon) > cutoff).sum().float(),
                sync_dist=True,
            )
            self.log(
                f"invert/fail_count_sample_{cutoff}",
                (torch.sum(abs(prior_samples - x_recon) > cutoff, dim=1) > 0).sum().float(),
                sync_dist=True,
            )
            x_pred = self.all_gather(x_pred).reshape(-1, *x_pred.shape[1:])
            fwd_logdets = self.all_gather(fwd_logdets).reshape(-1, *fwd_logdets.shape[1:])
            prior_log_q = self.all_gather(prior_log_q).reshape(-1, *prior_log_q.shape[1:])
            prior_samples = self.all_gather(prior_samples).reshape(-1, *prior_samples.shape[1:])

        log_q = prior_log_q.flatten() + fwd_logdets.flatten()

        return x_pred, log_q, prior_samples


if __name__ == "__main__":
    _ = NormalizingFlowLitModule(None, None, None, None)
