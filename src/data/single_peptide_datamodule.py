import logging
from pathlib import Path
from typing import Optional

import mdtraj as md
import numpy as np
import openmm
import openmm.app
import torch
import torchvision
from huggingface_hub import snapshot_download

from src.data.base_datamodule import BaseDataModule
from src.data.datasets.tensor_dataset import TensorDataset
from src.data.energy.openmm import OpenMMBridge, OpenMMEnergy
from src.data.preprocessing.tica import get_tica_model
from src.data.transforms.center_of_mass import CenterOfMassTransform
from src.data.transforms.rotation import Random3DRotationTransform
from src.data.transforms.standardize import StandardizeTransform


class SinglePeptideDataModule(BaseDataModule):
    def __init__(
        self,
        repo_id: str,
        data_dir: str,
        sequence: str,
        temperature: float,
        num_dimensions: int,
        num_atoms: int,
        com_augmentation: bool = False,
        num_eval_samples: int = 10_000,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

        self.repo_name = self.hparams.repo_id.split("/")[-1]
        self.trajectory_name = f"{self.hparams.sequence}_{self.hparams.temperature}K"

        # Setup paths
        self.trajectory_data_dir = f"{data_dir}/{self.repo_name}/{self.trajectory_name}"
        self.train_data_path = f"{self.trajectory_data_dir}/{self.trajectory_name}_train.npy"
        self.val_data_path = f"{self.trajectory_data_dir}/{self.trajectory_name}_val.npy"
        self.test_data_path = f"{self.trajectory_data_dir}/{self.trajectory_name}_test.npy"
        self.pdb_path = f"{self.trajectory_data_dir}/{self.trajectory_name}.pdb"

        # For compatibility with transferable case
        self.val_sequences = [self.hparams.sequence]
        self.test_sequences = [self.hparams.sequence]

    def prepare_data(self):
        """Download + preprocessing data. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        Path(f"{self.hparams.data_dir}/{self.repo_name}").mkdir(parents=True, exist_ok=True)

        local_dir = snapshot_download(
            repo_id=self.hparams.repo_id,
            repo_type="dataset",
            local_dir=f"{self.hparams.data_dir}/{self.repo_name}",
            allow_patterns=[f"{self.trajectory_name}/*"],
            use_auth_token=True,
        )
        logging.info(f"Downloaded dataset to {local_dir}")

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        # Divide batch size by the number of devices.
        if self.trainer is not None:
            if self.hparams.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.hparams.batch_size}) is not divisible by the number "
                    "of devices ({self.trainer.world_size}).",
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size
        else:
            self.batch_size_per_device = self.hparams.batch_size

        # Load the data
        data_train = np.load(self.train_data_path)
        data_val = np.load(self.val_data_path)
        data_test = np.load(self.test_data_path)

        # Tensorize the data
        data_train = torch.from_numpy(data_train)
        data_val = torch.from_numpy(data_val)
        data_test = torch.from_numpy(data_test)

        # Load the PDB file
        self.pdb = openmm.app.PDBFile(self.pdb_path)

        # Load the topology from the PDB file
        self.topology = md.load_topology(self.pdb_path)

        # For compatibility to transferable BG
        self.topology_dict = {self.hparams.sequence: self.topology}

        # Compute std on standardized data
        self.std = (data_train - data_train.mean(dim=1, keepdim=True)).std()

        transform_list = [
            StandardizeTransform(self.std),
            Random3DRotationTransform(),
        ]
        if self.hparams.com_augmentation:
            transform_list.append(CenterOfMassTransform())
        train_transforms = torchvision.transforms.Compose(transform_list)
        self.data_train = TensorDataset(
            data=data_train,
            transform=train_transforms,
        )

        test_transforms = StandardizeTransform(self.std)
        self.data_val = TensorDataset(
            data=data_val,
            transform=test_transforms,
        )
        self.data_test = TensorDataset(
            data=data_test,
            transform=test_transforms,
        )

        logging.info(f"Train dataset size: {len(self.data_train)}")
        logging.info(f"Validation dataset size: {len(self.data_val)}")
        logging.info(f"Test dataset size: {len(self.data_test)}")

    def setup_potential(self):
        """
        Set up the OpenMM potential energy function.

        Returns:
            OpenMMEnergy: An energy function wrapper around the OpenMM system and integrator.
        """
        if self.hparams.sequence in ["Ace-A-Nme", "Ace-AAA-Nme"]:
            forcefield = openmm.app.ForceField("amber99sbildn.xml", "tip3p.xml", "amber99_obc.xml")

            system = forcefield.createSystem(
                self.pdb.topology,
                nonbondedMethod=openmm.app.NoCutoff,
                nonbondedCutoff=0.9 * openmm.unit.nanometer,
                constraints=None,
            )
            temperature = 300
            integrator = openmm.LangevinMiddleIntegrator(
                temperature * openmm.unit.kelvin,
                0.3 / openmm.unit.picosecond
                if self.hparams.sequence == "Ace-AAA-Nme"
                else 1.0 / openmm.unit.picosecond,
                1.0 * openmm.unit.femtosecond,
            )
            potential = OpenMMEnergy(bridge=OpenMMBridge(system, integrator, platform_name="CUDA"))
        else:
            forcefield = openmm.app.ForceField("amber14-all.xml", "implicit/obc1.xml")
            temperature = 310

            system = forcefield.createSystem(
                self.pdb.topology,
                nonbondedMethod=openmm.app.CutoffNonPeriodic,
                nonbondedCutoff=2.0 * openmm.unit.nanometer,
                constraints=None,
            )
            integrator = openmm.LangevinMiddleIntegrator(
                temperature * openmm.unit.kelvin,
                0.3 / openmm.unit.picosecond,
                1.0 * openmm.unit.femtosecond,
            )

            # Initialize potential
            potential = OpenMMEnergy(bridge=OpenMMBridge(system, integrator, platform_name="CUDA"))

        return potential

    def prepare_eval(self, sequence: str, prefix: str):
        """
        Prepare evaluation data and energy function for validation or test trajectories.

        Selects trajectory data based on the provided prefix, constructs a TICA model,
        subsamples the trajectory, applies normalization, and sets up a potential energy
        function. Returns all components required for evaluation.

        Args:
            sequence (str): Unused compatibility argument for integration with
                TransferablePeptideDatamodule.
            prefix (str): Dataset split to evaluate on. Must be either "val" or "test".

        Returns:
            tuple: A 5-tuple containing:
                - true_samples (torch.Tensor): Normalized and subsampled trajectory samples.
                - permutations (None): Placeholder for compatibility, not used here.
                - encodings (None): Placeholder for compatibility, not used here.
                - energy_fn (Callable): Function mapping positions → energy values.
                - tica_model: Model with TICA projection parameters computed from the trajectory.
        """
        if prefix == "test":
            true_samples = self.data_test.data
        elif prefix == "val":
            true_samples = self.data_val.data
        else:
            raise ValueError(f"Unknown prefix: {prefix}. Use 'val' or 'test'.")

        tica_model = get_tica_model(true_samples, self.topology)

        # Subsample the true trajectory
        true_samples = true_samples[:: len(true_samples) // self.hparams.num_eval_samples]
        true_samples = self.normalize(true_samples)

        permutations = None
        encodings = None
        potential = self.setup_potential()
        energy_fn = lambda x: potential.energy(self.unnormalize(x)).flatten()

        return true_samples, permutations, encodings, energy_fn, tica_model
