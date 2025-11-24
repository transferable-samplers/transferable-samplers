from dataclasses import dataclass
import torch

@dataclass
class SamplesData:
    x: torch.Tensor
    proposal_energy: torch.Tensor
    target_energy: torch.Tensor
    importance_logits: torch.Tensor = None

    def __post_init__(self):
        assert len(self.x) == len(self.proposal_energy) == len(self.target_energy)
        if self.importance_logits is not None:
            assert len(self.x) == len(self.importance_logits)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return SamplerData(
            self.x[index],
            self.proposal_energy[index],
            self.target_energy[index],
            self.importance_logits[index],
            self.importance_logits[index] if self.importance_logits is not None else None,
        )

