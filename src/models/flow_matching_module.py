import copy
from functools import partial
from typing import Optional

import torch
from torchdyn.core import NeuralODE

from src.models.base_lightning_module import BaseLightningModule
from src.models.neural_networks.wrappers import TorchDynWrapper
from src.utils.dataclasses import SystemCond
from src.utils.pylogger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=False)


class FlowMatchingModule(BaseLightningModule):
    """Flow matching model."""

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        prior,
        sigma: float = 0.0,
        logp_tol_scale: float = 1.0,
        atol: float = 1e-5,
        rtol: float = 1e-5,
        mean_free_prior: bool = True,
        compile_net: bool = False,
        fix_symmetry: bool = True,
        drop_unfixable_symmetry: bool = False,
        output_dir: str = "",
        source_energy_config=None,
        train_from_buffer: bool = False,
    ) -> None:
        super().__init__(
            net=net,
            optimizer=optimizer,
            scheduler=scheduler,
            prior=prior,
            compile_net=compile_net,
            fix_symmetry=fix_symmetry,
            drop_unfixable_symmetry=drop_unfixable_symmetry,
            output_dir=output_dir,
            source_energy_config=source_energy_config,
            train_from_buffer=train_from_buffer,
        )
        self.sigma = sigma
        self.logp_tol_scale = logp_tol_scale
        self.atol = atol
        self.rtol = rtol
        self.nfe = 0
        self.num_integrations = 0

    def forward(self, t: torch.Tensor, x: torch.Tensor, encodings, mask) -> torch.Tensor:
        return self.net(t, x, encodings=encodings, node_mask=mask)

    def _get_xt(self, x0, x1, t, mask=None):
        mu_t = (1.0 - t) * x0 + t * x1

        if not self.sigma == 0.0:
            num_samples = x1.shape[0]
            num_tokens = x1.shape[1] // self.trainer.datamodule.num_dimensions
            noise = self.prior.sample(num_samples, num_tokens, mask=None, device=x1.device).flatten(start_dim=1)
            xt = mu_t + self.sigma * noise
            if mask is not None:
                xt = xt.view(num_samples, num_tokens, -1) * mask[..., None]
                xt = xt.view(num_samples, -1)
        else:
            xt = mu_t

        return xt

    def _get_flow_targets(self, x0, x1):
        vt_flow = x1 - x0
        return vt_flow

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        if self.train_from_buffer:
            batch = self._buffer.sample(batch["x"].shape[0])

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

    @torch.no_grad()
    def _integrate(self, net: torch.nn.Module, x: torch.Tensor, encodings=None, reverse=False, compute_dlogp=True) -> torch.Tensor:
        batch_size = x.shape[0]
        num_atoms = x.shape[1]

        x = x.reshape(batch_size, -1)  # Ensure x is 2D

        dlogp_init = torch.zeros((x.shape[0], 1), device=x.device)
        t_span = torch.linspace(1, 0, 2) if reverse else torch.linspace(0, 1, 2)

        eval_fn = partial(copy.deepcopy(net), encodings=encodings)

        wrapped_net = TorchDynWrapper(
            eval_fn,
            compute_dlogp=compute_dlogp,
            logp_tol_scale=self.logp_tol_scale,
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
            dlogp_out = x[..., -1] * self.logp_tol_scale
            x = x[..., :-1]
        else:
            dlogp_out = dlogp_init.squeeze(-1)
        x = x.reshape(batch_size, num_atoms, -1)

        return x, dlogp_out.view(-1)

    def proposal_energy(
        self, net: torch.nn.Module, x: torch.Tensor, system_cond: Optional[SystemCond] = None,
    ) -> torch.Tensor:
        encodings = system_cond.encodings if system_cond else None
        z_pred, dlogp_rev = self._integrate(net, x, encodings=encodings, reverse=True)
        # dlogp_rev is log|det(dx/dz)| = -log|det(dz/dx)|, so logq = logp_z - dlogp_rev
        logq = self.prior.logp(z_pred) - dlogp_rev
        return -logq  # energy is negative log probability

    def generate_proposal(
        self, net: torch.nn.Module, num_samples: int,
        system_cond: Optional[SystemCond] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encodings = system_cond.encodings if system_cond else None

        if encodings is None:
            num_atoms = self.trainer.datamodule.num_atoms
        else:
            num_atoms = encodings["atom_type"].size(0)

        z = self.prior.sample(num_samples, num_atoms, device=self.device)
        logp_z = self.prior.logp(z)

        if encodings is not None:
            encodings = {
                key: tensor.unsqueeze(0).repeat(num_samples, 1).to(self.device)
                for key, tensor in encodings.items()
            }

        with torch.no_grad():
            x_pred, dlogp = self._integrate(net, z, encodings=encodings, reverse=False)

        logq = logp_z + dlogp

        return x_pred, -logq
