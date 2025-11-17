import glob
import logging
import os
import pickle
from typing import Optional, Union

import numpy as np
import openmm
import openmm.app
import torch
import torchvision
from omegaconf import ListConfig, OmegaConf

from src.data.base_datamodule import BaseDataModule
from src.data.datasets.buffer import ReplayBuffer
from src.data.datasets.peptides_dataset import PeptidesDatasetWithBuffer, build_peptides_dataset
from src.data.datasets.webdataset import build_webdataset
from src.data.energy.openmm import OpenMMBridge, OpenMMEnergy
from src.data.preprocessing.preprocessing import (
    prepare_and_cache_encodings_dict,
    prepare_and_cache_pdb_dict,
    prepare_and_cache_permutations_dict,
    prepare_and_cache_topology_dict,
)
from src.data.preprocessing.tica import TicaModel
from src.data.transforms.add_encodings import AddEncodingsTransform
from src.data.transforms.add_permutations import AddPermutationsTransform
from src.data.transforms.center_of_mass import CenterOfMassTransform
from src.data.transforms.padding import PaddingTransform
from src.data.transforms.rotation import Random3DRotationTransform
from src.data.transforms.standardize import StandardizeTransform
from src.utils.huggingface import download_and_extract_pdb_tarfiles, download_evaluation_data


class TransferablePeptideDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir: str,
        num_aa_min: int,
        num_aa_max: int,
        num_dimensions: int,
        num_atoms: int,
        precomputed_std: float,
        hf_repo_id: str = "transferable-samplers/many-peptides-md",
        wds_repo_path: str = "webdatasets/single_frames",
        com_augmentation: bool = False,
        num_eval_samples: int = 10000,
        val_sequences: Union[str, list[str]] = None,
        test_sequences: Union[str, list[str]] = None,
        buffer: ReplayBuffer = None,
        buffer_ckpt_path: str = None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

        self.pdb_dir = os.path.join(data_dir, "pdbs")

        self.wds_cache_dir = os.path.join(data_dir, wds_repo_path)

        self.evaluation_data_path = os.path.join(data_dir, "trajectories_subsampled")
        self.val_data_path = os.path.join(self.evaluation_data_path, "val")
        self.test_data_path = os.path.join(self.evaluation_data_path, "test")

        self.preproc_cache_dir = os.path.join(data_dir, "preprocessing_cache")
        self.pdb_dict_pkl_path = os.path.join(self.preproc_cache_dir, "pdb_dict.pkl")
        self.topology_dict_pkl_path = os.path.join(self.preproc_cache_dir, "topology_dict.pkl")
        self.encodings_dict_pkl_path = os.path.join(self.preproc_cache_dir, "encodings_dict.pkl")
        self.permutations_dict_pkl_path = os.path.join(self.preproc_cache_dir, "permutations_dict.pkl")

        self.buffer = buffer
        self.buffer_ckpt_path = buffer_ckpt_path

        # Parse evaluation sequence hparam (which sequences to use for val / testing)
        if isinstance(self.hparams.val_sequences, str):
            self.val_sequences = [self.hparams.val_sequences]
        elif isinstance(self.hparams.val_sequences, ListConfig):
            self.val_sequences = OmegaConf.to_container(self.hparams.val_sequences)
        else:
            raise TypeError("Unrecognized type for val_sequences")
        if isinstance(self.hparams.test_sequences, str):
            self.test_sequences = [self.hparams.test_sequences]
        elif isinstance(self.hparams.test_sequences, ListConfig):
            self.test_sequences = OmegaConf.to_container(self.hparams.test_sequences)
        else:
            raise TypeError("Unrecognized type for test_sequences")

        # Precomputed std for standardization
        self.std = torch.tensor(self.hparams.precomputed_std)

    def prepare_data(self) -> None:
        """Download + preprocessing data. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """

        os.makedirs(self.wds_cache_dir, exist_ok=True)
        os.makedirs(self.preproc_cache_dir, exist_ok=True)

        download_and_extract_pdb_tarfiles(self.hparams.data_dir)
        download_evaluation_data(self.hparams.data_dir)

        # Check if the preprocessing cache already exists
        if not all(
            os.path.exists(p)
            for p in [
                self.pdb_dict_pkl_path,
                self.topology_dict_pkl_path,
                self.encodings_dict_pkl_path,
                self.permutations_dict_pkl_path,
            ]
        ):
            logging.info("Preparing and caching PDB, topology, encodings, and permutations dicts.")
            pdb_paths = glob.glob(os.path.join(self.pdb_dir, "*", "*.pdb"))
            # Do data preprocessinging here and cache the results - to be loaded by workers later.
            pdb_dict = prepare_and_cache_pdb_dict(pdb_paths, self.pdb_dict_pkl_path, delimiter=".")
            topology_dict = prepare_and_cache_topology_dict(pdb_dict, self.topology_dict_pkl_path)
            _ = prepare_and_cache_encodings_dict(topology_dict, self.encodings_dict_pkl_path)
            _ = prepare_and_cache_permutations_dict(topology_dict, self.permutations_dict_pkl_path)
        else:
            logging.info("PDB, topology, encodings, permutation dicts already cached")

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
                    f"Batch size ({self.hparams.batch_size}) is not divisible by "
                    f"the number of devices ({self.trainer.world_size}).",
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # Load cached data preprocessing dict
        with open(self.pdb_dict_pkl_path, "rb") as f:
            self.pdb_dict = pickle.load(f)  # noqa: S301
        logging.info(f"Loaded pdb dict from {self.pdb_dict_pkl_path}")
        with open(self.topology_dict_pkl_path, "rb") as f:
            self.topology_dict = pickle.load(f)  # noqa: S301
        logging.info(f"Loaded topology dict from {self.topology_dict_pkl_path}")
        with open(self.encodings_dict_pkl_path, "rb") as f:
            self.encodings_dict = pickle.load(f)  # noqa: S301
        logging.info(f"Loaded encodings dict from {self.encodings_dict_pkl_path}")
        with open(self.permutations_dict_pkl_path, "rb") as f:
            self.permutations_dict = pickle.load(f)  # noqa: S301
        logging.info(f"Loaded permutations dict from {self.permutations_dict_pkl_path}")

        # Build train transformations pipeline
        train_transform_list = [
            StandardizeTransform(self.std),
            Random3DRotationTransform(),
        ]
        if self.hparams.com_augmentation:
            train_transform_list.append(CenterOfMassTransform())
        train_transform_list = train_transform_list + [
            AddEncodingsTransform(self.encodings_dict),
            AddPermutationsTransform(self.permutations_dict),
            PaddingTransform(self.hparams.num_atoms),
        ]
        train_transforms = torchvision.transforms.Compose(train_transform_list)

        # Resume buffer if needed
        if self.buffer_ckpt_path is not None:
            if os.path.exists(self.buffer_ckpt_path):
                logging.info(f"Resuming buffer from checkpoint: {self.buffer_ckpt_path}")
                self.buffer.load(self.buffer_ckpt_path)
                logging.info(f"Number of Samples Loaded: {len(self.buffer)}")
            else:
                logging.info(f"Buffer checkpoint path {self.buffer_ckpt_path} not found! Ignoring...")

        if self.buffer is None:
            # Build training webdataset
            self.data_train = build_webdataset(
                repo_id=self.hparams.hf_repo_id,
                repo_path=self.hparams.wds_repo_path,
                cache_dir=self.wds_cache_dir,
                num_aa_min=self.hparams.num_aa_min,
                num_aa_max=self.hparams.num_aa_max,
                transform=train_transforms,
            )
        else:
            assert len(self.test_sequences) == 1, "Can currently only self-refine on one system at a time."
            self.data_train = PeptidesDatasetWithBuffer(
                buffer=self.buffer,
                transform=train_transforms,
            )

        self.data_val = build_peptides_dataset(
            path=self.val_data_path,
            num_aa_min=self.hparams.num_aa_min,
            num_aa_max=self.hparams.num_aa_max,
            transform=train_transforms,
        )

        test_transform_list = [
            StandardizeTransform(self.std),
            AddEncodingsTransform(self.encodings_dict),
            AddPermutationsTransform(self.permutations_dict),
            PaddingTransform(self.hparams.num_atoms),
        ]

        test_transforms = torchvision.transforms.Compose(test_transform_list)

        self.data_test = build_peptides_dataset(
            path=self.test_data_path,
            num_aa_min=self.hparams.num_aa_min,
            num_aa_max=self.hparams.num_aa_max,
            transform=test_transforms,
        )

    def setup_potential(self, sequence: str):
        """
        Set up an OpenMM potential energy function for a given peptide sequence.

        Args:
            sequence (str): Peptide sequence identifier corresponding to a loaded PDB structure.

        Returns:
            OpenMMEnergy: An energy function wrapper around the OpenMM system and integrator.
        """
        forcefield = openmm.app.ForceField("amber14-all.xml", "implicit/obc1.xml")
        nonbondedMethod = openmm.app.CutoffNonPeriodic
        nonbondedCutoff = 2.0 * openmm.unit.nanometer
        temperature = 310

        # Initialize forcefield systemq
        system = forcefield.createSystem(
            self.pdb_dict[sequence].topology,
            nonbondedMethod=nonbondedMethod,
            nonbondedCutoff=nonbondedCutoff,
            constraints=None,
        )

        # Initialize integrator
        integrator = openmm.LangevinMiddleIntegrator(
            temperature * openmm.unit.kelvin,
            0.3 / openmm.unit.picosecond,
            1.0 * openmm.unit.femtosecond,
        )

        # Initialize potential
        platform_name = "CUDA" if torch.cuda.is_available() else "CPU"
        potential = OpenMMEnergy(bridge=OpenMMBridge(system, integrator, platform_name=platform_name))

        return potential

    def prepare_eval(self, sequence: str, prefix: str = None):
        """
        Prepare evaluation data and energy function for a given peptide sequence.

        Loads the subsampled trajectory data (positions and TICA projection) from disk,
        applies normalization, retrieves permutations and encodings, and constructs
        the potential energy function. Returns all components required for evaluation.

        Args: sequence (str): Peptide sequence identifier to prepare evaluation data for.
            prefix (str): Unused compatibility argument for integration with
                SinglePeptideDatamodule.

        Returns:
            tuple: A 5-tuple containing:
                - true_samples (torch.Tensor): Normalized trajectory samples.
                - permutations (Any): Permutations associated with the sequence.
                - encodings (Any): Encodings associated with the sequence.
                - energy_fn (Callable): Function mapping positions → energy values.
                - tica_model (TicaModel): Model with TICA projection parameters.
        """
        subsampled_trajectory_npz = np.load(
            os.path.join(self.val_data_path, f"{len(sequence)}AA", f"{sequence}_subsampled.npz")
            if sequence in self.val_sequences
            else os.path.join(self.test_data_path, f"{len(sequence)}AA", f"{sequence}_subsampled.npz"),
        )

        true_samples = torch.from_numpy(subsampled_trajectory_npz["positions"])
        tica_model = TicaModel(
            projection=subsampled_trajectory_npz["tica_projection"],
            mean=subsampled_trajectory_npz["tica_mean"],
            dim=subsampled_trajectory_npz["tica_dim"],
        )

        true_samples = self.normalize(true_samples)
        permutations = self.permutations_dict[sequence]
        encodings = self.encodings_dict[sequence]
        potential = self.setup_potential(sequence)
        energy_fn = lambda x: potential.energy(self.unnormalize(x)).flatten()

        return true_samples, permutations, encodings, energy_fn, tica_model

    def save_buffer(self):
        """
        Save the training buffer checkpoint to disk, if a path is configured.

        Uses the buffer checkpoint path defined in the object to persist
        the current training buffer state.

        Returns:
            None
        """
        if self.buffer_ckpt_path is not None:
            logging.info(f"Saving Buffer: {self.buffer_ckpt_path}")
            self.data_train.buffer.save(self.buffer_ckpt_path)
