from dataclasses import dataclass
import torch
from typing import Callable, Dict, Tuple

import glob
import logging
import os
import pickle
from dataclasses import dataclass
from typing import Callable, Optional, Union

import numpy as np
import openmm
import openmm.app
import torch
import torchvision
from omegaconf import ListConfig, OmegaConf

from src.data.preprocessing.tica import TicaModel

@dataclass
class ProposalModel:
    sample_fn: Callable  # (num_samples, batch_size, system_cond) -> (samples, log_q, extra)
    energy_fn: Callable  # (samples) -> proposal_energy

    def sample(
        self,
        num_samples: int,
        batch_size: int,
        system_cond: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.sample_fn(num_samples, batch_size, system_cond)

    def energy(self, x: torch.Tensor) -> torch.Tensor:
        return self.energy_fn(x)


@dataclass
class SystemConditioning:
    """Inputs needed for model inference."""
    permutations: dict
    encodings: dict


@dataclass
class EvaluationInputs:
    """Inputs needed for evaluation metrics."""
    true_samples: torch.Tensor
    tica_model: TicaModel
    topology: object  # mdtraj.Topology
    num_eval_samples: int


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
        return SamplesData(
            self.x[index],
            self.proposal_energy[index],
            self.target_energy[index],
            self.importance_logits[index] if self.importance_logits is not None else None,
        )