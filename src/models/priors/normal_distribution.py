import torch
from torch.distributions import Normal


class NormalDistribution:
    def __init__(self, num_dimensions: int = 3, mean: float = 0.0, std: float = 1.0, mean_free: bool = False):
        self.num_dimensions = num_dimensions
        self.mean = mean
        self.std = std
        self.distribution = Normal(mean, std)
        self.mean_free = mean_free

    def sample(self, num_samples: int, num_atoms: int, mask: torch.Tensor | None = None, device="cpu") -> torch.Tensor:
        x = self.distribution.sample((num_samples, num_atoms, self.num_dimensions)).to(device)
        if self.mean_free:
            if mask is None:
                mask = torch.ones((num_samples, num_atoms), device=device)
            com = (x * mask[..., None]).sum(dim=1, keepdims=True) / mask.sum(dim=1, keepdims=True)[..., None]
            x = x - com
            x *= mask[..., None]
        return x.reshape(num_samples, num_atoms, self.num_dimensions)

    def energy(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        assert x.dim() == 3
        num_samples = x.shape[0]
        num_atoms = x.shape[1]
        if mask is None:
            mask = torch.ones((num_samples, num_atoms), device=x.device)
        if self.mean_free:
            com = (x * mask[..., None]).sum(dim=1, keepdims=True) / mask.sum(dim=1, keepdims=True)[..., None]
            x = x - com
            x *= mask[..., None]

        pointwise_energy = -self.distribution.log_prob(x)

        pointwise_energy = pointwise_energy * mask.unsqueeze(-1)
        pointwise_energy = pointwise_energy.reshape(num_samples, -1)
        num_atoms = mask.sum(dim=-1, keepdim=True)
        # account for the pad tokens when taking the mean
        energy = pointwise_energy.sum(dim=-1, keepdims=True) / (num_atoms * self.num_dimensions)

        return energy
