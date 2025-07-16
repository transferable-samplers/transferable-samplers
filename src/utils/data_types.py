from dataclasses import dataclass

import torch



@dataclass
class SamplesData:
    samples: torch.Tensor
    energy: torch.Tensor
    logits: torch.Tensor = None

    def __post_init__(self):
        assert len(self.samples) == len(self.energy)
        if self.logits is not None:
            assert len(self.samples) == len(self.logits)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return SamplesData(
            self.samples[index],
            self.energy[index],
            self.logits[index] if self.logits is not None else None,
        )
