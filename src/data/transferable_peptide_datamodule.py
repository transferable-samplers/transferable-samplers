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

from src.data.base_datamodule import BaseDataModule
from src.data.datasets.buffer import ReplayBuffer
from src.data.datasets.peptides_dataset import PeptideDatasetWithBuffer, build_peptide_dataset
from src.data.datasets.webdataset import build_webdataset
from src.data.energy.openmm import OpenMMBridge, OpenMMEnergy
from src.data.preprocessing.preprocessing import load_preprocessing_cache, prepare_preprocessing_cache
from src.data.preprocessing.tica import TicaModel
from src.data.transforms.add_encodings import AddEncodingsTransform
from src.data.transforms.add_permutations import AddPermutationsTransform
from src.data.transforms.center_of_mass import CenterOfMassTransform
from src.data.transforms.padding import PaddingTransform
from src.data.transforms.rotation import Random3DRotationTransform
from src.data.transforms.standardize import StandardizeTransform
from src.utils.huggingface import download_and_extract_pdb_tarfiles, download_evaluation_data


from src.utils.dataclasses import SystemConditioning, EvaluationInputs


class TransferablePeptideDataModule(BaseDataModule):
    # Hardcoded HuggingFace repository configuration
    HF_REPO_ID = "transferable-samplers/many-peptides-md"
    WDS_REPO_PATH = "webdatasets/single_frames"
    
    def __init__(
        self,
        data_dir: str,
        num_aa_min: int,
        num_aa_max: int,
        num_dimensions: int,
        num_atoms: int,
        precomputed_std: float,
        com_augmentation: bool = False,
        num_eval_samples: int = 10000,
        val_sequences: Union[str, list[str]] = None,
        test_sequences: Union[str, list[str]] = None,
        buffer: ReplayBuffer = None,
        buffer_ckpt_path: str = None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        task_name: str = None,
    ):
        super().__init__(batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory)

        self.pdb_dir = os.path.join(data_dir, "pdbs")

        self.wds_cache_dir = os.path.join(data_dir, self.WDS_REPO_PATH)

        self.evaluation_data_path = os.path.join(data_dir, "trajectories_subsampled")
        self.val_data_path = os.path.join(self.evaluation_data_path, "val")
        self.test_data_path = os.path.join(self.evaluation_data_path, "test")

        self.preproc_cache_dir = os.path.join(data_dir, "preprocessing_cache")
        
        # Single cache file per subset
        self.train_cache_path = os.path.join(self.preproc_cache_dir, "train_cache.pkl")
        self.val_cache_path = os.path.join(self.preproc_cache_dir, "val_cache.pkl")
        self.test_cache_path = os.path.join(self.preproc_cache_dir, "test_cache.pkl")

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

        # Even with the cache the training cache is quite slow to load, so we avoid this for eval tasks
        self.requires_setup_train = task_name is not None and task_name != "eval"

    def prepare_data(self) -> None:
        """Download + preprocessing data. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """

        os.makedirs(self.wds_cache_dir, exist_ok=True)
        os.makedirs(self.preproc_cache_dir, exist_ok=True)

        # TODO - uncomment this once merged the new downloads
        # download_and_extract_pdb_tarfiles(self.hparams.data_dir)
        download_evaluation_data(self.hparams.data_dir)

        # Discover PDB files for each subset
        train_pdb_paths = glob.glob(os.path.join(self.pdb_dir, "train", "*.pdb"))
        val_pdb_paths = glob.glob(os.path.join(self.pdb_dir, "val", "*.pdb"))
        test_pdb_paths = glob.glob(os.path.join(self.pdb_dir, "test", "*.pdb"))
        
        logging.info(f"Found {len(train_pdb_paths)} train PDB files")
        logging.info(f"Found {len(val_pdb_paths)} val PDB files")
        logging.info(f"Found {len(test_pdb_paths)} test PDB files")

        # Prepare train cache from train PDB files
        prepare_preprocessing_cache(
            pdb_paths=train_pdb_paths,
            cache_path=self.train_cache_path,
        )

        # Prepare val cache from val PDB files
        prepare_preprocessing_cache(
            pdb_paths=val_pdb_paths,
            cache_path=self.val_cache_path,
        )

        # Prepare test cache from test PDB files
        prepare_preprocessing_cache(
            pdb_paths=test_pdb_paths,
            cache_path=self.test_cache_path,
        )

    def setup_train(self) -> None:
        """Setup training data and preprocessing."""
        # Load train preprocessing cache
        logging.info("Loading train preprocessing cache...")
        (
            self.train_pdb_dict,
            self.train_topology_dict,
            self.train_encodings_dict,
            self.train_permutations_dict,
        ) = load_preprocessing_cache(self.train_cache_path)
        logging.info(f"Loaded train cache: {len(self.train_pdb_dict)} sequences")

        # Build train transformations pipeline
        train_transform_list = [
            StandardizeTransform(self.std),
            Random3DRotationTransform(),
        ]
        if self.hparams.com_augmentation:
            train_transform_list.append(CenterOfMassTransform())
        train_transform_list = train_transform_list + [
            AddEncodingsTransform(self.train_encodings_dict),
            AddPermutationsTransform(self.train_permutations_dict),
            PaddingTransform(self.hparams.num_atoms),
        ]
        train_transforms = torchvision.transforms.Compose(train_transform_list)

        # Build training webdataset
        self.data_train = build_webdataset(
            repo_id=self.HF_REPO_ID,
            repo_path=self.WDS_REPO_PATH,
            cache_dir=self.wds_cache_dir,
            num_aa_min=self.hparams.num_aa_min,
            num_aa_max=self.hparams.num_aa_max,
            transform=train_transforms,
        )

    def setup_eval(self, split: str) -> None:
        """
        Setup evaluation data for a specific split.

        Args:
            split: Either "val" or "test".
        """
        assert split in ["val", "test"], f"split must be 'val' or 'test', got {split}"
        
        # Get split-specific paths and attributes
        if split == "val":
            cache_path = self.val_cache_path
            data_path = self.val_data_path
        else:  # test
            cache_path = self.test_cache_path
            data_path = self.test_data_path
        
        # Load preprocessing cache
        logging.info(f"Loading {split} preprocessing cache...")
        (
            pdb_dict,
            topology_dict,
            encodings_dict,
            permutations_dict,
        ) = load_preprocessing_cache(cache_path)
        logging.info(f"Loaded {split} cache: {len(pdb_dict)} sequences")
        
        # Store in split-specific attributes
        if split == "val":
            self.val_pdb_dict = pdb_dict
            self.val_topology_dict = topology_dict
            self.val_encodings_dict = encodings_dict
            self.val_permutations_dict = permutations_dict
        else:
            self.test_pdb_dict = pdb_dict
            self.test_topology_dict = topology_dict
            self.test_encodings_dict = encodings_dict
            self.test_permutations_dict = permutations_dict
        
        # Build transformations pipeline
        transform_list = [
            StandardizeTransform(self.std),
            AddEncodingsTransform(encodings_dict),
            AddPermutationsTransform(permutations_dict),
            PaddingTransform(self.hparams.num_atoms),
        ]
        transforms = torchvision.transforms.Compose(transform_list)
        
        # Discover NPZ files
        npz_file_paths = glob.glob(os.path.join(data_path, "*/*.npz"))
        logging.info(f"Found {len(npz_file_paths)} NPZ files in {data_path}")
        
        # Build dataset
        dataset = build_peptide_dataset(
            file_paths=npz_file_paths,
            num_aa_min=self.hparams.num_aa_min,
            num_aa_max=self.hparams.num_aa_max,
            transform=transforms,
        )
        
        # Store in split-specific attribute
        if split == "val":
            self.data_val = dataset
        else:
            self.data_test = dataset

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
                    f"the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.hparams.batch_size // self.trainer.world_size

        # Setup train, val, and test data
        if self.requires_setup_train:
            self.setup_train()
        else:
            self.data_train = None
        self.setup_eval("val")
        self.setup_eval("test")

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

        # Get PDB from the appropriate split cache
        if sequence in self.val_pdb_dict:
            pdb = self.val_pdb_dict[sequence]
        elif sequence in self.test_pdb_dict:
            pdb = self.test_pdb_dict[sequence]
        else:
            pdb = self.train_pdb_dict[sequence]

        # Initalize forcefield systemq
        system = forcefield.createSystem(
            pdb.topology,
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

    def prepare_eval(self, sequence: str, prefix: str = None) -> tuple[SystemConditioning, EvaluationInputs, Callable]:
        """
        Prepare evaluation data and energy function for a given peptide sequence.

        Loads the subsampled trajectory data (positions and TICA projection) from disk,
        applies normalization, retrieves permutations and encodings, and constructs
        the potential energy function. Returns all components required for evaluation.

        Args:
            sequence (str): Peptide sequence identifier to prepare evaluation data for.
            prefix (str): Unused compatibility argument for integration with
                SinglePeptideDatamodule.

        Returns:
            tuple: A 3-tuple containing:
                - system_cond (SystemConditioning): Permutations and encodings for model.
                - evaluation_inputs (EvaluationInputs): True samples and TICA model for evaluation.
                - target_energy_fn (Callable): Function mapping positions → energy values.
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
        
        # Get permutations and encodings from the appropriate split cache
        if sequence in self.val_sequences:
            permutations = self.val_permutations_dict[sequence]
            encodings = self.val_encodings_dict[sequence]
            topology = self.val_topology_dict[sequence]
        else:
            permutations = self.test_permutations_dict[sequence]
            encodings = self.test_encodings_dict[sequence]
            topology = self.test_topology_dict[sequence]
        
        potential = self.setup_potential(sequence)
        target_energy_fn = lambda x: potential.energy(self.unnormalize(x)).flatten()

        system_cond = SystemConditioning(permutations=permutations, encodings=encodings)
        evaluation_inputs = EvaluationInputs(
            true_samples=true_samples,
            tica_model=tica_model,
            topology=topology,
            num_eval_samples=self.hparams.num_eval_samples,
        )

        return system_cond, evaluation_inputs, target_energy_fn
