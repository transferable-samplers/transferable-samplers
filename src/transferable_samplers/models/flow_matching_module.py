from __future__ import annotations

import copy
from functools import partial
from typing import Any

import torch
from torchdyn.core import NeuralODE

from transferable_samplers.models.base_lightning_module import BaseLightningModule
from transferable_samplers.models.priors.prior import Prior
from transferable_samplers.nn.wrappers import TorchDynWrapper
from transferable_samplers.utils.dataclasses import SourceEnergyConfig, SystemCond
from transferable_samplers.utils.pylogger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=False)


class FlowMatchingModule(BaseLightningModule):
    """Flow matching model."""

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        prior: Prior,
        scheduler: Any = None,
        compile_net: bool = False,
        source_energy_config: SourceEnergyConfig | None = None,
        train_from_buffer: bool = False,
        mean_free_prior: bool = True,
        sigma: float = 0.0,
        dlogp_tol_scale: float = 1.0,
        atol: float = 1e-5,
        rtol: float = 1e-5,
    ) -> None:
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

        # store hyperparams/config
        self.sigma = sigma
        self.dlogp_tol_scale = dlogp_tol_scale
        self.atol = atol
        self.rtol = rtol

        # set runtime state
        self.nfe = 0
        self.num_integrations = 0

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        if self.train_from_buffer:
            # pyrefly: ignore [missing-attribute]
            batch = self._buffer.sample(batch["x"].shape[0], device=self.device)

        assert len(batch["x"].shape) == 3, "molecules must be a pointcloud (batch_size, num_atoms, 3)"

        x1 = batch["x"]

        num_samples = x1.shape[0]
        num_tokens = x1.shape[1]

        encodings = batch.get("encodings", None)
        mask = batch.get("mask", None)

        t = torch.rand(num_samples, 1, device=x1.device)
        z = self.prior.sample(num_samples, num_tokens, mask, device=x1.device)

        x1 = x1.flatten(start_dim=1)
        z = z.flatten(start_dim=1)

        xt = self._get_xt(z, x1, t, mask)
        vt_flow = self._get_flow_targets(z, x1)

        vt_pred = self.forward(t, xt, encodings=encodings, mask=mask)

        assert len(vt_pred.shape) == 2
        assert len(vt_flow.shape) == 2

        loss = torch.sum((vt_pred - vt_flow) ** 2, dim=-1)
        if mask is not None:
            loss = loss / mask.int().sum(-1)
        loss = loss.mean()

        batch_value = self.train_metrics(loss)
        self.log_dict(batch_value, prog_bar=True)
        return loss

    def generate_proposal(
        self,
        net: torch.nn.Module,
        num_samples: int,
        system_cond: SystemCond | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encodings = system_cond.encodings if system_cond else None

        if encodings is None:
            # pyrefly: ignore [missing-attribute]
            num_atoms = self.trainer.datamodule.num_atoms
        else:
            num_atoms = encodings["atom_type"].size(0)

        z = self.prior.sample(num_samples, num_atoms, device=self.device)
        logp_z = self.prior.logp(z)

        batched_cond = system_cond.for_batch(num_samples, self.device) if system_cond else None
        encodings = batched_cond.encodings if batched_cond else None

        x_pred, dlogp = self._integrate(net, z, encodings=encodings, reverse=False)

        logq = logp_z + dlogp

        return x_pred, -logq

    def proposal_energy(
        self,
        net: torch.nn.Module,
        x: torch.Tensor,
        system_cond: SystemCond | None = None,
    ) -> torch.Tensor:
        if torch.is_grad_enabled() and x.requires_grad:
            logger.warning(
                "We have not tested differentiation of FlowMatchingModule.proposal_energy()."
                "Please test well if using this in a differentiable context!"
            )
        encodings = system_cond.encodings if system_cond else None
        z_pred, dlogp_rev = self._integrate(net, x, encodings=encodings, reverse=True)
        # dlogp_rev is log|det(dx/dz)| = -log|det(dz/dx)|, so logq = logp_z - dlogp_rev
        logq = self.prior.logp(z_pred) - dlogp_rev
        return -logq  # energy is negative log probability

    # pyrefly: ignore [bad-override]
    def forward(
        self, t: torch.Tensor, x: torch.Tensor, encodings: dict[str, torch.Tensor] | None, mask: torch.Tensor | None
    ) -> torch.Tensor:
        return self.net(t, x, encodings=encodings, node_mask=mask)

    def _integrate(
        self,
        net: torch.nn.Module,
        x: torch.Tensor,
        encodings: dict[str, torch.Tensor] | None = None,
        reverse: bool = False,
        compute_dlogp: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        num_atoms = x.shape[1]

        x = x.reshape(batch_size, -1)  # Ensure x is 2D

        dlogp_init = torch.zeros((x.shape[0], 1), device=x.device)
        t_span = torch.linspace(1, 0, 2) if reverse else torch.linspace(0, 1, 2)

        eval_fn = partial(copy.deepcopy(net), encodings=encodings)

        wrapped_net = TorchDynWrapper(
            eval_fn,
            compute_dlogp=compute_dlogp,
            dlogp_tol_scale=self.dlogp_tol_scale,
        )

        node = NeuralODE(
            wrapped_net,
            atol=self.atol,
            rtol=self.rtol,
            solver="dopri5",
            sensitivity="adjoint",
        )
        if compute_dlogp:
            x = torch.cat([x, dlogp_init], dim=-1)
        x = node.trajectory(x, t_span=t_span)[-1]
        logger.info(f"nfe: {wrapped_net.nfe}")
        self.nfe += wrapped_net.nfe
        self.num_integrations += 1
        wrapped_net.nfe = 0
        if compute_dlogp:
            dlogp_out = x[..., -1] * self.dlogp_tol_scale
            x = x[..., :-1]
        else:
            dlogp_out = dlogp_init.squeeze(-1)
        x = x.reshape(batch_size, num_atoms, -1)

        # pyrefly: ignore [bad-return]
        return x, dlogp_out.view(-1)

    def _get_xt(
        self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        mu_t = (1.0 - t) * x0 + t * x1

        if not self.sigma == 0.0:
            num_samples = x1.shape[0]
            # pyrefly: ignore [missing-attribute]
            num_tokens = x1.shape[1] // self.trainer.datamodule.num_dimensions
            noise = self.prior.sample(num_samples, num_tokens, mask=None, device=x1.device).flatten(start_dim=1)
            xt = mu_t + self.sigma * noise
            if mask is not None:
                xt = xt.view(num_samples, num_tokens, -1) * mask[..., None]
                xt = xt.view(num_samples, -1)
        else:
            xt = mu_t

        return xt

    def _get_flow_targets(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        vt_flow = x1 - x0
        return vt_flow
