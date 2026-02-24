import copy
from functools import partial
from typing import Optional

import torch
from torchdyn.core import NeuralODE

from src.models.base_lightning_module import BaseLightningModule
from src.models.neural_networks.wrappers import TorchdynWrapper, torch_wrapper
from src.utils import pylogger
from src.utils.dataclasses import SystemCond

logger = pylogger.RankedLogger(__name__, rank_zero_only=False)


class FlowMatchLitModule(BaseLightningModule):
    """Flow matching model."""

    def __init__(self, sigma=0.0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.nfe = 0
        self.num_integrations = 0
        self.eps = 1e-1
        if "strict_loading" in kwargs:
            self.strict_loading = kwargs["strict_loading"]

    def forward(self, t: torch.Tensor, x: torch.Tensor, encodings, mask) -> torch.Tensor:
        return self.net(t, x, encodings=encodings, node_mask=mask)

    def get_xt(self, x0, x1, t, mask=None):
        mu_t = (1.0 - t) * x0 + t * x1

        if not self.hparams.sigma == 0.0:
            num_samples = x1.shape[0]
            num_tokens = x1.shape[1] // self.trainer.datamodule.hparams.num_dimensions
            noise = self.prior.sample(num_samples, num_tokens, mask=None, device=x1.device).flatten(start_dim=1)
            xt = mu_t + self.hparams.sigma * noise
            if mask is not None:
                xt = xt.view(num_samples, num_tokens, -1) * mask[..., None]
                xt = xt.view(num_samples, -1)
        else:
            xt = mu_t

        return xt

    def get_flow_targets(self, x0, x1):
        vt_flow = x1 - x0
        return vt_flow

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        if self.train_from_buffer:
            batch = self._buffer.sample(batch["x"].shape[0])

        if self.hparams.use_distill_loss:
            raise NotImplementedError("Distillation loss not implemented for flow matching.")

        assert len(batch["x"].shape) == 3, "molecules must be a pointcloud (batch_size, num_atoms, 3)"

        x1 = batch["x"]

        num_samples = x1.shape[0]
        num_tokens = x1.shape[1]

        encodings = batch.get("encodings", None)
        mask = batch.get("mask", None)

        t = torch.rand(num_samples, 1, device=x1.device)
        prior_samples = self.prior.sample(num_samples, num_tokens, mask, device=x1.device)

        x1 = x1.flatten(start_dim=1)
        prior_samples = prior_samples.flatten(start_dim=1)

        xt = self.get_xt(prior_samples, x1, t, mask)
        vt_flow = self.get_flow_targets(prior_samples, x1)

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
    def flow(self, net: torch.nn.Module, x: torch.Tensor, encodings=None, reverse=False, dummy_ll=False) -> torch.Tensor:
        batch_size = x.shape[0]
        num_atoms = x.shape[1]

        x = x.reshape(batch_size, -1)  # Ensure x is 2D

        dlog_p = torch.zeros((x.shape[0], 1), device=x.device)
        t_span = torch.linspace(1, 0, 2) if reverse else torch.linspace(0, 1, 2)

        eval_fn = partial(copy.deepcopy(net), encodings=encodings)

        if self.hparams.div_estimator == "ito":
            x_ito, dlog_p_ito = self.sde_integrate(x, reverse=reverse)
            return x_ito, dlog_p_ito

        if dummy_ll:
            wrapped_net = torch_wrapper(eval_fn)
            logger.info("Using dummy ll")
        else:
            wrapped_net = TorchdynWrapper(
                eval_fn,
                div_estimator=self.hparams.div_estimator,
                logp_tol_scale=self.hparams.logp_tol_scale,
                n_eps=self.hparams.n_eps,
            )
            logger.info(f"Using {self.hparams.div_estimator} with n_eps {self.hparams.n_eps}")

        node = NeuralODE(
            wrapped_net,
            atol=self.hparams.atol,
            rtol=self.hparams.rtol,
            solver="dopri5",
            sensitivity="adjoint",
        )
        if not dummy_ll:
            x = torch.cat([x, dlog_p], dim=-1)
        x = node.trajectory(x, t_span=t_span)[-1]
        logger.info(f"nfe: {wrapped_net.nfe}")
        self.nfe += wrapped_net.nfe
        self.num_integrations += 1
        wrapped_net.nfe = 0
        if not dummy_ll:
            dlog_p = x[..., -1] * self.hparams.logp_tol_scale
            x = x[..., :-1]
        x = x.reshape(batch_size, num_atoms, -1)

        return x, dlog_p

    def euler_maruyama_step(
        self,
        t: torch.Tensor,
        x: torch.Tensor,
        dt: float,
        step: int,
        batch_size: int,
    ):
        vt = self.net(t, x, d_base=None)

        sigma_t_squared = 2 * (1 - t) / torch.clip(t, min=self.eps)
        sigma_t_squared = 2 * (1 - t) / torch.clip(t, min=0.1)
        sigma_t = sigma_t_squared**0.5

        # st is correct we checked
        st = vt + sigma_t_squared * (t * vt - x) / torch.clip(1 - t, min=self.eps) / 2
        eps_t = torch.randn_like(x)
        noise_t = sigma_t * eps_t * (dt**0.5)
        dxt = st * dt + noise_t

        score_t = (t * vt - x) / torch.clip(1 - t, min=self.eps)
        a = -x.shape[-1] / torch.clip(t, min=self.eps) * dt
        b = (sigma_t_squared / 2 * score_t * score_t).sum(-1) * dt
        c = (score_t * noise_t).sum(-1)

        dlogp = a + b + c
        # Update the state
        x_next = x + dxt
        return x_next, dlogp

    def sde_integrate(self, x0: torch.Tensor, reverse=False) -> torch.Tensor:
        batch_size = x0.shape[0]
        time_range = 1.0
        start_time = 1.0 if reverse else 0.0
        end_time = 1.0 - start_time
        if self.num_integrations == 0:
            num_integration_steps = 1000
        else:
            num_integration_steps = self.num_integrations
        times = torch.linspace(start_time, end_time, num_integration_steps + 1, device=x0.device)[:-1]
        x = x0

        x0.requires_grad = True
        samples = []
        dlogp_sum = 0

        for step, t in enumerate(times):
            x, dlogp = self.euler_maruyama_step(t, x, time_range / num_integration_steps, step, batch_size)
            dlogp_sum += dlogp
            samples.append(x)

        samples = torch.stack(samples)
        return x, dlogp_sum

    def proposal_energy(
        self, net: torch.nn.Module, x: torch.Tensor, system_cond: Optional[SystemCond] = None,
    ) -> torch.Tensor:
        encodings = system_cond.encodings if system_cond else None
        x, dlogp = self.flow(net, x, encodings=encodings, reverse=True)
        return -(-self.prior.energy(x).view(-1) - dlogp.view(-1))

    def sample_proposal(
        self, net: torch.nn.Module, num_samples: int,
        system_cond: Optional[SystemCond] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        encodings = system_cond.encodings if system_cond else None

        if encodings is None:
            num_atoms = self.trainer.datamodule.hparams.num_atoms
        else:
            num_atoms = encodings["atom_type"].size(0)

        data_dim = num_atoms * self.trainer.datamodule.hparams.num_atoms

        prior_samples = self.prior.sample(num_samples, num_atoms, device=self.device)
        prior_log_p = -self.prior.energy(prior_samples) * data_dim

        if encodings is not None:
            encodings = {
                key: tensor.unsqueeze(0).repeat(num_samples, 1).to(self.device)
                for key, tensor in encodings.items()
            }

        with torch.no_grad():
            samples, dlog_p = self.flow(net, prior_samples, encodings=encodings, reverse=False)

        log_p = prior_log_p.flatten() + dlog_p.flatten()

        return samples, log_p
