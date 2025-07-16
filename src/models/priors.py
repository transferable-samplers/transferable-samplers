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


if __name__ == "__main__":
    normal_dist = NormalDistribution(num_dimensions=3, mean=0.0, std=1.0, mean_free=True)
    samples = normal_dist.sample(num_samples=10, num_atoms=8)
    print("Samples", samples)
    print("Samples shape", samples.shape)

    num_pad = 2
    zero_pad = torch.zeros((samples.shape[0], 3 * num_pad))
    samples_padded = torch.concat([samples, zero_pad], dim=-1)
    mask = torch.ones((samples.shape[0], samples.shape[-1] // 3))
    mask = torch.concat([mask, torch.zeros((samples.shape[0], num_pad))], dim=-1)

    energy = normal_dist.energy(samples)
    energy_padded = normal_dist.energy(samples_padded, mask=mask)

    energy_error = abs(energy - energy_padded).sum()
    print(f"Energy Error between unpad and padded: {energy_error}")
    assert torch.allclose(energy, energy_padded, atol=1e-8), f"Energy Error: {energy_error}"

    # test center of mass with mask
    def com_test():
        num_atoms = 10
        num_pad = 2
        x = torch.randn((16, num_atoms, 3))
        random_pad = torch.randn((16, num_pad, 3))

        x_padded = torch.concat([x, random_pad], dim=1)
        mask = torch.ones((16, num_atoms))
        mask = torch.concat([mask, torch.zeros((16, num_pad))], dim=-1)

        com = x.mean(dim=1, keepdims=True)
        com_padded = (x_padded * mask[..., None]).sum(dim=1, keepdims=True) / mask.sum(dim=1, keepdims=True)[..., None]

        com_error = (abs(com - com_padded)).sum()
        print(f"COM error: {com_error}")
        assert com.shape == com_padded.shape
        assert torch.allclose(com, com_padded, atol=1e-8), "com and com_padded do not match"

        x -= com
        x_padded -= com_padded
        x_padded *= mask[..., None]
        x_padded = x_padded[:, :num_atoms, :]

        print(f"x and x_padded error: {abs(x - x_padded).sum()}")
        assert torch.allclose(x, x_padded)

        samples = normal_dist.sample(num_samples=16, num_atoms=12, mask=mask)
        samples = samples.reshape(16, num_atoms + num_pad, 3)
        com = samples[:, :num_atoms].mean(dim=1, keepdims=True)
        print(f"COM: {com}")
        assert torch.allclose(com, torch.zeros_like(com), atol=1e-7), "COM is not zero after sampling with mask"

    com_test()
