from __future__ import annotations

from pathlib import Path

import mdtraj as md
import numpy as np
import openmm
import openmm.app
import openmm.unit
import torch
import torchvision
from huggingface_hub import snapshot_download

from transferable_samplers.data.base_datamodule import BaseDataModule
from transferable_samplers.data.datasets.dummy_dataset import DummyDataset
from transferable_samplers.data.datasets.tensor_dataset import TensorDataset
from transferable_samplers.data.energy.openmm_energy import OpenMMEnergy
from transferable_samplers.data.preprocessing.tica import get_tica_model
from transferable_samplers.data.transforms.center_of_mass import CenterOfMassTransform
from transferable_samplers.data.transforms.rotation import Random3DRotationTransform
from transferable_samplers.data.transforms.standardize import StandardizeTransform
from transferable_samplers.utils.dataclasses import EvalContext, SamplesData, TargetEnergy
from transferable_samplers.utils.pylogger import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class SinglePeptideDataModule(BaseDataModule):
    """Datamodule for a single fixed peptide sequence.

    Downloads trajectory data from HuggingFace Hub, computes standardization
    statistics from the training split, and builds an OpenMM energy function
    for evaluation.

    Args:
        sequence: Amino acid sequence string (e.g. ``"Ace-AAA-Nme"``).
        temperature: Simulation temperature in Kelvin.

    See ``BaseDataModule`` for remaining args.
    """

    HF_REPO_ID = "transferable-samplers/sequential-boltzmann-generators-data"

    def __init__(
        self,
        data_dir: str,
        num_dimensions: int,
        num_atoms: int,
        batch_size: int = 64,
        num_workers: int = 0,
        persistent_workers: bool = False,
        pin_memory: bool = False,
        com_augmentation: bool = False,
        num_eval_samples: int = 10_000,
        train_from_buffer: bool = False,
        sequence: str = "",
        temperature: float = 310.0,
    ) -> None:
        super().__init__(
            data_dir=data_dir,
            num_dimensions=num_dimensions,
            num_atoms=num_atoms,
            batch_size=batch_size,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            pin_memory=pin_memory,
            com_augmentation=com_augmentation,
            num_eval_samples=num_eval_samples,
            train_from_buffer=train_from_buffer,
        )

        self.sequence = sequence
        self.temperature = temperature

        # Construct derived paths
        self.repo_name = self.HF_REPO_ID.split("/")[-1]
        self.trajectory_name = f"{self.sequence}_{self.temperature}K"
        self.trajectory_data_dir = f"{data_dir}/{self.repo_name}/{self.trajectory_name}"
        self.train_data_path = f"{self.trajectory_data_dir}/{self.trajectory_name}_train.npy"
        self.val_data_path = f"{self.trajectory_data_dir}/{self.trajectory_name}_val.npy"
        self.test_data_path = f"{self.trajectory_data_dir}/{self.trajectory_name}_test.npy"
        self.pdb_path = f"{self.trajectory_data_dir}/{self.trajectory_name}.pdb"

        # For compatibility with transferable case
        self.val_sequences = [self.sequence]
        self.test_sequences = [self.sequence]

    def prepare_data(self) -> None:
        """Download trajectory data from HuggingFace Hub if not already present.

        Lightning ensures this is called only within a single process on CPU,
        so downloading logic is safe here. In multi-node training, execution
        depends on ``self.prepare_data_per_node()``.

        Do not use it to assign state (``self.x = y``).
        """
        required_files = [self.train_data_path, self.val_data_path, self.test_data_path, self.pdb_path]
        if all(Path(f).exists() for f in required_files):
            log.info(f"All required files already exist for {self.trajectory_name}, skipping download.")
            return

        Path(f"{self.data_dir}/{self.repo_name}").mkdir(parents=True, exist_ok=True)

        local_dir = snapshot_download(
            repo_id=self.HF_REPO_ID,
            repo_type="dataset",
            local_dir=f"{self.data_dir}/{self.repo_name}",
            allow_patterns=[f"{self.trajectory_name}/*"],
            token=True,
        )
        log.info(f"Downloaded dataset to {local_dir}")

    def setup(self, stage: str | None = None) -> None:
        """Load trajectory data and compute standardization statistics.

        Called by Lightning before ``trainer.fit()``, ``trainer.validate()``,
        ``trainer.test()``, and ``trainer.predict()``. A barrier after
        ``prepare_data`` ensures all processes wait until data is ready.

        Args:
            stage: One of ``"fit"``, ``"validate"``, ``"test"``, or ``"predict"``.
        """
        self._validate_and_set_batch_size()

        # Dummy val/test datasets — actual evaluation happens via prepare_eval + callbacks
        self.data_val = DummyDataset()
        self.data_test = DummyDataset()

        # Load the data (needed to compute std even in eval-only runs)
        data_train = np.load(self.train_data_path)
        data_train = torch.from_numpy(data_train)

        # Compute std on centered data
        self.std = (data_train - data_train.mean(dim=1, keepdim=True)).std()

        if stage != "fit":
            return

        # Load the PDB file
        self.pdb = openmm.app.PDBFile(self.pdb_path)

        # Load the topology from the PDB file
        self.topology = md.load_topology(self.pdb_path)

        # For compatibility with transferable BG
        self.topology_dict = {self.sequence: self.topology}

        transform_list = [
            # pyrefly: ignore [bad-argument-type]
            StandardizeTransform(self.std),
            Random3DRotationTransform(),
        ]
        if self.com_augmentation:
            # pyrefly: ignore [bad-argument-type]
            transform_list.append(CenterOfMassTransform())
        train_transforms = torchvision.transforms.Compose(transform_list)

        if self.train_from_buffer:
            # Buffer transforms are the same as train transforms for single peptide
            # (no encoding/permutation lookups or padding needed)
            # Stored to be accessed by the model for buffer sampling.
            # pyrefly: ignore [missing-attribute]
            self.buffer_transforms = self.train_transforms

            # Placeholder dataset; model owns the buffer and samples from it
            # pyrefly: ignore [missing-attribute]
            assert isinstance(self.trainer.limit_train_batches, int), (
                "trainer.limit_train_batches must be set to an integer when using "
                "train_from_buffer (the model needs to know how many batches to sample from the buffer each epoch)."
            )
            limit = self.trainer.limit_train_batches
            dummy_size = limit * self.batch_size_per_device
            self.data_train = DummyDataset(size=dummy_size)
        else:
            # pyrefly: ignore [bad-assignment]
            self.data_train = TensorDataset(
                data=data_train,
                transform=train_transforms,
            )

        log.info(f"Train dataset size: {len(self.data_train)}")

    def prepare_eval(self, sequence: str, stage: str) -> EvalContext:
        """Prepare evaluation data and energy function for validation or test trajectories."""
        assert sequence == self.sequence, (
            f"Requested eval sequence '{sequence}' does not match datamodule sequence '{self.sequence}'"
        )
        if stage == "test":
            true_samples = torch.from_numpy(np.load(self.test_data_path))
        elif stage == "val":
            true_samples = torch.from_numpy(np.load(self.val_data_path))
        else:
            raise ValueError(f"Unknown stage: {stage}. Use 'val' or 'test'.")

        # Load PDB/topology if not already loaded (e.g. eval-only runs)
        if not hasattr(self, "pdb"):
            self.pdb = openmm.app.PDBFile(self.pdb_path)
        if not hasattr(self, "topology"):
            self.topology = md.load_topology(self.pdb_path)
        if not hasattr(self, "std"):
            data_train = torch.from_numpy(np.load(self.train_data_path))
            self.std = (data_train - data_train.mean(dim=1, keepdim=True)).std()

        tica_model = get_tica_model(true_samples, self.topology)

        # Subsample the true trajectory
        true_samples = true_samples[:: len(true_samples) // self.num_eval_samples]

        potential = self._setup_potential()
        energy_fn = lambda x: potential(x)

        true_data = SamplesData(
            samples=true_samples,
            E_target=potential(true_samples),
        )

        return EvalContext(
            true_data=true_data,
            target_energy=TargetEnergy(energy_fn=energy_fn, normalization_std=self.std),
            normalization_std=self.std,
            system_cond=None,
            tica_model=tica_model,
            topology=self.topology,
        )

    def _setup_potential(self) -> OpenMMEnergy:
        """Set up the OpenMM potential energy function.

        Returns:
            OpenMMEnergy: An energy function wrapper around the OpenMM system and integrator.
        """
        if self.sequence in ["Ace-A-Nme", "Ace-AAA-Nme"]:
            forcefield = openmm.app.ForceField("amber99sbildn.xml", "tip3p.xml", "amber99_obc.xml")

            system = forcefield.createSystem(
                self.pdb.topology,
                nonbondedMethod=openmm.app.NoCutoff,
                # pyrefly: ignore [missing-attribute]
                nonbondedCutoff=0.9 * openmm.unit.nanometer,
                constraints=None,
            )
            temperature = 300
            # pyrefly: ignore [missing-attribute]
            integrator = openmm.LangevinMiddleIntegrator(
                # pyrefly: ignore [unsupported-operation]
                temperature * openmm.unit.kelvin,
                # pyrefly: ignore [missing-attribute]
                0.3 / openmm.unit.picosecond if self.sequence == "Ace-AAA-Nme" else 1.0 / openmm.unit.picosecond,
                # pyrefly: ignore [missing-attribute]
                1.0 * openmm.unit.femtosecond,
            )
            platform_name = "CUDA" if torch.cuda.is_available() else "CPU"
            device_index = torch.cuda.current_device() if torch.cuda.is_available() else None
            potential = OpenMMEnergy(system, integrator, platform_name=platform_name, device_index=device_index)
        else:
            forcefield = openmm.app.ForceField("amber14-all.xml", "implicit/obc1.xml")
            temperature = 310

            system = forcefield.createSystem(
                self.pdb.topology,
                nonbondedMethod=openmm.app.CutoffNonPeriodic,
                # pyrefly: ignore [missing-attribute]
                nonbondedCutoff=2.0 * openmm.unit.nanometer,
                constraints=None,
            )
            # pyrefly: ignore [missing-attribute]
            integrator = openmm.LangevinMiddleIntegrator(
                # pyrefly: ignore [unsupported-operation]
                temperature * openmm.unit.kelvin,
                # pyrefly: ignore [missing-attribute]
                0.3 / openmm.unit.picosecond,
                # pyrefly: ignore [missing-attribute]
                1.0 * openmm.unit.femtosecond,
            )

            # Initialize potential
            platform_name = "CUDA" if torch.cuda.is_available() else "CPU"
            device_index = torch.cuda.current_device() if torch.cuda.is_available() else None
            potential = OpenMMEnergy(system, integrator, platform_name=platform_name, device_index=device_index)

        return potential
