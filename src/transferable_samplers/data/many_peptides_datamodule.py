import glob
import logging
import os
import pickle
from typing import Optional, Union

import mdtraj as md
import numpy as np
import openmm
import openmm.app
import torch
import torchvision
from omegaconf import ListConfig, OmegaConf

from transferable_samplers.data.base_datamodule import BaseDataModule
from transferable_samplers.data.datasets.dummy_dataset import DummyDataset
from transferable_samplers.data.datasets.webdataset import build_webdataset
from transferable_samplers.data.energy.openmm_energy import OpenMMEnergy
from transferable_samplers.data.preprocessing.encodings import get_encodings
from transferable_samplers.data.preprocessing.permutations import get_permutations_dict
from transferable_samplers.data.preprocessing.cache import (
    prepare_and_cache_encodings_dict,
    prepare_and_cache_pdb_dict,
    prepare_and_cache_permutations_dict,
    prepare_and_cache_topology_dict,
)
from transferable_samplers.data.preprocessing.tica import TicaModel
from transferable_samplers.data.transforms.add_encodings import AddEncodingsTransform
from transferable_samplers.data.transforms.add_permutations import AddPermutationsTransform
from transferable_samplers.data.transforms.center_of_mass import CenterOfMassTransform
from transferable_samplers.data.transforms.padding import PaddingTransform
from transferable_samplers.data.transforms.rotation import Random3DRotationTransform
from transferable_samplers.data.transforms.standardize import StandardizeTransform
from transferable_samplers.utils.dataclasses import EvalContext, SamplesData, SystemCond, TargetEnergy
from transferable_samplers.utils.huggingface import download_and_extract_pdb_tarfiles, download_evaluation_data


class ManyPeptidesDataModule(BaseDataModule):
    HF_REPO_ID = "transferable-samplers/many-peptides-md"
    WDS_REPO_PATH = "webdatasets/single_frames"
    NUM_TARFILES = 5000

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
        precomputed_std: float = 1.0,
        val_sequences: Union[str, list[str], None] = None,
        test_sequences: Union[str, list[str], None] = None,
        num_aa_min: int = 2,
        num_aa_max: int = 8,
        system_cond_ids: list[str] | None = None,
    ):
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

        # Validate/normalize inputs
        if val_sequences is None:
            self.val_sequences = []
        elif isinstance(val_sequences, str):
            self.val_sequences = [val_sequences]
        elif isinstance(val_sequences, (list, ListConfig)):
            self.val_sequences = OmegaConf.to_container(val_sequences) if isinstance(val_sequences, ListConfig) else val_sequences
        else:
            raise TypeError("Unrecognized type for val_sequences")
        if test_sequences is None:
            self.test_sequences = []
        elif isinstance(test_sequences, str):
            self.test_sequences = [test_sequences]
        elif isinstance(test_sequences, (list, ListConfig)):
            self.test_sequences = OmegaConf.to_container(test_sequences) if isinstance(test_sequences, ListConfig) else test_sequences
        else:
            raise TypeError("Unrecognized type for test_sequences")

        self.precomputed_std = precomputed_std
        self.num_aa_min = num_aa_min
        self.num_aa_max = num_aa_max
        self.system_cond_ids = system_cond_ids or []

        # Construct derived paths
        self.pdb_dir = os.path.join(data_dir, "pdbs")
        self.wds_cache_dir = os.path.join(data_dir, self.WDS_REPO_PATH)
        self.evaluation_data_path = os.path.join(data_dir, "trajectories_subsampled")
        self.val_data_path = os.path.join(self.evaluation_data_path, "val")
        self.test_data_path = os.path.join(self.evaluation_data_path, "test")
        self.preproc_cache_dir = os.path.join(data_dir, "preprocessing_cache")
        self.pdb_dict_pkl_path = os.path.join(self.preproc_cache_dir, "pdb_dict.pkl")
        self.topology_dict_pkl_path = os.path.join(self.preproc_cache_dir, "topology_dict.pkl")
        self.encodings_dict_pkl_path = os.path.join(self.preproc_cache_dir, "encodings_dict.pkl")
        self.permutations_dict_pkl_path = os.path.join(self.preproc_cache_dir, "permutations_dict.pkl")

        # Precomputed std for standardization
        self.std = torch.tensor(self.precomputed_std)

    def prepare_data(self) -> None:
        """Download + preprocessing data. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """

        os.makedirs(self.wds_cache_dir, exist_ok=True)
        os.makedirs(self.preproc_cache_dir, exist_ok=True)

        download_and_extract_pdb_tarfiles(self.data_dir)
        download_evaluation_data(self.data_dir)

        # Build or load cached preprocessing dicts (each function handles caching internally)
        pdb_paths = glob.glob(os.path.join(self.pdb_dir, "*", "*.pdb"))
        pdb_dict = prepare_and_cache_pdb_dict(pdb_paths, self.pdb_dict_pkl_path, delimiter=".")
        topology_dict = prepare_and_cache_topology_dict(pdb_dict, self.topology_dict_pkl_path)

        if "encodings" in self.system_cond_ids:
            prepare_and_cache_encodings_dict(topology_dict, self.encodings_dict_pkl_path)
        if "permutations" in self.system_cond_ids:
            prepare_and_cache_permutations_dict(topology_dict, self.permutations_dict_pkl_path)

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        self._validate_and_set_batch_size()

        # Dummy val/test datasets — actual evaluation happens via prepare_eval + callbacks
        self.data_val = DummyDataset()
        self.data_test = DummyDataset()

        if stage != "fit":
            return

        # Both WebDataset and buffer training require knowing the number of batches in advance
        assert isinstance(self.trainer.limit_train_batches, int), (
            "trainer.limit_train_batches must be set to an integer for wds and buffer training"
        )

        # Load cached data preprocessing dicts
        with open(self.pdb_dict_pkl_path, "rb") as f:
            self.pdb_dict = pickle.load(f)  # noqa: S301
        logging.info(f"Loaded pdb dict from {self.pdb_dict_pkl_path}")
        with open(self.topology_dict_pkl_path, "rb") as f:
            self.topology_dict = pickle.load(f)  # noqa: S301
        logging.info(f"Loaded topology dict from {self.topology_dict_pkl_path}")

        self.encodings_dict = None
        if "encodings" in self.system_cond_ids:
            with open(self.encodings_dict_pkl_path, "rb") as f:
                self.encodings_dict = pickle.load(f)  # noqa: S301
            logging.info(f"Loaded encodings dict from {self.encodings_dict_pkl_path}")

        self.permutations_dict = None
        if "permutations" in self.system_cond_ids:
            with open(self.permutations_dict_pkl_path, "rb") as f:
                self.permutations_dict = pickle.load(f)  # noqa: S301
            logging.info(f"Loaded permutations dict from {self.permutations_dict_pkl_path}")

        # Build transformations pipeline
        base_transform_list = [
            StandardizeTransform(self.std),
            Random3DRotationTransform(),
        ]
        if self.com_augmentation:
            base_transform_list.append(CenterOfMassTransform())

        if self.train_from_buffer:
            # Buffer transforms: batchable geometric transforms.
            # Stored to be accessed by the model for buffer sampling.
            buffer_transform_list = list(base_transform_list)
            self.buffer_transforms = torchvision.transforms.Compose(buffer_transform_list)

            # Placeholder dataset; model owns the buffer and samples from it
            dummy_size = self.trainer.limit_train_batches * self.batch_size_per_device
            self.data_train = DummyDataset(size=dummy_size)

        else:
            # Full train transforms: also include encoding/permutation lookups for webdataset
            train_transform_list = list(base_transform_list)
            if self.encodings_dict is not None:
                train_transform_list.append(AddEncodingsTransform(self.encodings_dict))
            if self.permutations_dict is not None:
                train_transform_list.append(AddPermutationsTransform(self.permutations_dict))
            train_transform_list.append(PaddingTransform(self.num_atoms))
            train_transforms = torchvision.transforms.Compose(train_transform_list)

            tar_urls = self._resolve_tar_urls()
            self.data_train = build_webdataset(
                tar_urls=tar_urls,
                cache_dir=self.wds_cache_dir,
                num_aa_min=self.num_aa_min,
                num_aa_max=self.num_aa_max,
                transform=train_transforms,
            )

    def prepare_eval(self, sequence: str, stage: str = None) -> EvalContext:
        """Prepare evaluation data and energy function for a given peptide sequence.

        Loads PDB directly from disk — no cache loading since it can be slow and
        memory-intensive for small evaluation runs.

        Args:
            sequence: Peptide sequence identifier to prepare evaluation data for.
            stage: Dataset split ("val" or "test"). Used to select data path.

        Returns:
            EvalContext with all components required for evaluation.
        """
        is_val = sequence in self.val_sequences
        is_test = sequence in self.test_sequences
        assert is_val or is_test, f"Sequence {sequence} not found in val or test sequences."
        assert stage == ("val" if is_val else "test"), f"Stage '{stage}' does not match sequence '{sequence}' (found in {'val' if is_val else 'test'})."

        data_path = self.val_data_path if is_val else self.test_data_path
        subsampled_trajectory_npz = np.load(
            os.path.join(data_path, f"{len(sequence)}AA", f"{sequence}_subsampled.npz"),
        )

        true_samples = torch.from_numpy(subsampled_trajectory_npz["positions"])
        tica_model = TicaModel(
            projection=subsampled_trajectory_npz["tica_projection"],
            mean=subsampled_trajectory_npz["tica_mean"],
            dim=subsampled_trajectory_npz["tica_dim"],
        )

        pdb = self._load_pdb(sequence, stage)
        topology = md.Topology.from_openmm(pdb.topology)

        potential = self._setup_potential(pdb)
        energy_fn = lambda x: potential(x)

        system_cond = None
        if "encodings" in self.system_cond_ids or "permutations" in self.system_cond_ids:
            encodings = get_encodings(topology) if "encodings" in self.system_cond_ids else None
            permutations = get_permutations_dict({sequence: topology})[sequence] if "permutations" in self.system_cond_ids else None
            system_cond = SystemCond(
                permutations=permutations,
                encodings=encodings,
            )

        true_data = SamplesData(
            samples=true_samples,
            E_target=potential(true_samples),
        )

        return EvalContext(
            true_data=true_data,
            target_energy=TargetEnergy(energy_fn=energy_fn, normalization_std=self.std),
            normalization_std=self.std,
            system_cond=system_cond,
            tica_model=tica_model,
            topology=topology,
        )

    def _check_cache_for_tarfiles(self) -> list[str] | None:
        """Check if all expected tar files are cached locally.

        Returns:
            List of local tar file paths if all expected tars are present, None otherwise.
        """
        local_tars = sorted(glob.glob(os.path.join(self.wds_cache_dir, "*.tar")))
        if len(local_tars) >= self.NUM_TARFILES:
            logging.info(f"Found {len(local_tars)} cached tar files in {self.wds_cache_dir}, skipping HuggingFace.")
            return local_tars
        if local_tars:
            logging.info(
                f"Found {len(local_tars)} cached tar files but expected {self.NUM_TARFILES}. "
                "Falling back to HuggingFace."
            )
        return None

    def _resolve_hf_tar_urls(self) -> list[str]:
        hf_url_prefix = (
            f"https://huggingface.co/datasets/"
            f"{self.HF_REPO_ID}/resolve/main/{self.WDS_REPO_PATH}"
        )
        return [f"{hf_url_prefix}/{i:04d}.tar" for i in range(self.NUM_TARFILES)]

    def _resolve_tar_urls(self) -> list[str]:
        """Resolve tar file URLs, preferring local cache over HuggingFace."""
        local_tars = self._check_cache_for_tarfiles()
        if local_tars is not None:
            return local_tars
        urls = self._resolve_hf_tar_urls()
        logging.info(f"Using WebDataset cache dir: {self.wds_cache_dir}")
        return urls

    def _load_pdb(self, sequence: str, stage: str):
        """Load PDB file for a sequence directly from disk.

        Args:
            sequence: Peptide sequence identifier.
            stage: Dataset split ("val" or "test"), used to locate the PDB subdirectory.
        """
        pdb_path = os.path.join(self.pdb_dir, stage, f"{sequence}.pdb")
        if not os.path.exists(pdb_path):
            raise FileNotFoundError(f"No PDB file found at {pdb_path}")
        return openmm.app.PDBFile(pdb_path)

    def _setup_potential(self, pdb):
        """Set up an OpenMM potential energy function from a PDB object."""
        forcefield = openmm.app.ForceField("amber14-all.xml", "implicit/obc1.xml")
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=openmm.app.CutoffNonPeriodic,
            nonbondedCutoff=2.0 * openmm.unit.nanometer,
            constraints=None,
        )
        integrator = openmm.LangevinMiddleIntegrator(
            310 * openmm.unit.kelvin,
            0.3 / openmm.unit.picosecond,
            1.0 * openmm.unit.femtosecond,
        )
        platform_name = "CUDA" if torch.cuda.is_available() else "CPU"
        device_index = torch.cuda.current_device() if torch.cuda.is_available() else None
        return OpenMMEnergy(system, integrator, platform_name=platform_name, device_index=device_index)
