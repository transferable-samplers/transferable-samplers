from dataclasses import dataclass

import torch

from src.utils.dist_utils import all_gather_cat


@dataclass
class SMCParticles:
    """State of SMC particles during annealing.

    Bundles positions, log-weights, lineage tracking, and the source/target
    energies and gradients needed by the Langevin proposal and MH steps.
    """
    x: torch.Tensor              # (batch, atoms, 3)
    logw: torch.Tensor            # (batch,)
    lineage: torch.Tensor         # (batch,) — index of the original proposal each particle descends from
    E_source: torch.Tensor        # (batch,)
    E_target: torch.Tensor        # (batch,)
    E_source_grad: torch.Tensor   # (batch, atoms, 3)
    E_target_grad: torch.Tensor   # (batch, atoms, 3)

    def __post_init__(self):
        batch = self.x.shape[0]
        assert self.logw.shape == (batch,), f"logw shape {self.logw.shape}, expected ({batch},)"
        assert self.lineage.shape == (batch,), f"lineage shape {self.lineage.shape}, expected ({batch},)"
        assert self.E_source.shape == (batch,), f"E_source shape {self.E_source.shape}, expected ({batch},)"
        assert self.E_target.shape == (batch,), f"E_target shape {self.E_target.shape}, expected ({batch},)"
        assert self.E_source_grad.shape == self.x.shape, f"E_source_grad shape {self.E_source_grad.shape}, expected {self.x.shape}"
        assert self.E_target_grad.shape == self.x.shape, f"E_target_grad shape {self.E_target_grad.shape}, expected {self.x.shape}"

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return SMCParticles(
            x=self.x[index],
            logw=self.logw[index],
            lineage=self.lineage[index],
            E_source=self.E_source[index],
            E_target=self.E_target[index],
            E_source_grad=self.E_source_grad[index],
            E_target_grad=self.E_target_grad[index],
        )


def merge_particles(mask: torch.Tensor, true_particles: SMCParticles, false_particles: SMCParticles) -> SMCParticles:
    """Merge two SMCParticles using a boolean mask.

    Where mask is True, take from true_particles; where False, take from false_particles.
    Preserves the original batch ordering.
    """
    batch_size = len(mask)
    device = mask.device

    x = torch.empty(batch_size, *false_particles.x.shape[1:], device=device)
    logw = torch.empty(batch_size, device=device)
    lineage = torch.empty(batch_size, dtype=false_particles.lineage.dtype, device=device)
    E_source = torch.empty(batch_size, device=device)
    E_target = torch.empty(batch_size, device=device)
    E_source_grad = torch.empty(batch_size, *false_particles.E_source_grad.shape[1:], device=device)
    E_target_grad = torch.empty(batch_size, *false_particles.E_target_grad.shape[1:], device=device)

    x[mask] = true_particles.x
    x[~mask] = false_particles.x
    logw[mask] = true_particles.logw
    logw[~mask] = false_particles.logw
    lineage[mask] = true_particles.lineage
    lineage[~mask] = false_particles.lineage
    E_source[mask] = true_particles.E_source
    E_source[~mask] = false_particles.E_source
    E_target[mask] = true_particles.E_target
    E_target[~mask] = false_particles.E_target
    E_source_grad[mask] = true_particles.E_source_grad
    E_source_grad[~mask] = false_particles.E_source_grad
    E_target_grad[mask] = true_particles.E_target_grad
    E_target_grad[~mask] = false_particles.E_target_grad

    return SMCParticles(
        x=x, logw=logw, lineage=lineage,
        E_source=E_source, E_target=E_target,
        E_source_grad=E_source_grad, E_target_grad=E_target_grad,
    )


def all_gather_particles(particles: SMCParticles) -> SMCParticles:
    """All-gather every field of SMCParticles across DDP ranks."""
    return SMCParticles(
        x=all_gather_cat(particles.x),
        logw=all_gather_cat(particles.logw),
        lineage=all_gather_cat(particles.lineage),
        E_source=all_gather_cat(particles.E_source),
        E_target=all_gather_cat(particles.E_target),
        E_source_grad=all_gather_cat(particles.E_source_grad),
        E_target_grad=all_gather_cat(particles.E_target_grad),
    )
