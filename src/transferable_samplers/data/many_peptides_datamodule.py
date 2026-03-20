from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import mdtraj as md
import numpy as np
import openmm
import openmm.app
import openmm.unit
import torch
import torchvision
from omegaconf import ListConfig, OmegaConf

from transferable_samplers.data.base_datamodule import BaseDataModule
from transferable_samplers.data.datasets.dummy_dataset import DummyDataset
from transferable_samplers.data.datasets.webdataset import build_webdataset
from transferable_samplers.data.energy.openmm_energy import OpenMMEnergy
from transferable_samplers.data.preprocessing.cache import (
    prepare_and_cache_encodings_dict,
    prepare_and_cache_pdb_dict,
    prepare_and_cache_permutations_dict,
    prepare_and_cache_topology_dict,
)
from transferable_samplers.data.preprocessing.encodings import get_encodings
from transferable_samplers.data.preprocessing.permutations import get_permutations_dict
from transferable_samplers.data.preprocessing.tica import TicaModel
from transferable_samplers.data.transforms.add_encodings import AddEncodingsTransform
from transferable_samplers.data.transforms.add_permutations import AddPermutationsTransform
from transferable_samplers.data.transforms.center_of_mass import CenterOfMassTransform
from transferable_samplers.data.transforms.padding import PaddingTransform
from transferable_samplers.data.transforms.rotation import Random3DRotationTransform
from transferable_samplers.data.transforms.standardize import StandardizeTransform
from transferable_samplers.utils.dataclasses import EvalContext, SamplesData, SystemCond, TargetEnergy
from transferable_samplers.utils.huggingface import download_and_extract_pdb_tarfiles, download_evaluation_data
from transferable_samplers.utils.pylogger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=True)


class ManyPeptidesDataModule(BaseDataModule):
    """Datamodule for transferable training / evaluation on ManyPeptidesMD dataset.

    Training data is streamed from WebDataset tar archives (cached locally or
    fetched from HuggingFace Hub). Evaluation data uses pre-subsampled
    trajectories with precomputed TICA projections.

    System conditioning (encodings and/or permutations) is controlled by
    ``system_cond_ids`` and cached as pickle files during ``prepare_data``.

    Args:
        precomputed_std: Precomputed standardization std across all sequences.
        val_sequences: Sequence(s) to evaluate during validation.
        test_sequences: Sequence(s) to evaluate during testing.
        num_aa_min: Minimum amino acid count for WebDataset filtering.
        num_aa_max: Maximum amino acid count for WebDataset filtering.
        system_cond_ids: Which conditioning to compute (``"encodings"``,
            ``"permutations"``, or both).

    See ``BaseDataModule`` for remaining args.
    """

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
        val_sequences: str | list[str] | None = None,
        test_sequences: str | list[str] | None = None,
        num_aa_min: int = 2,
        num_aa_max: int = 8,
        system_cond_ids: list[str] | None = None,
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

        # Validate/normalize inputs
        if val_sequences is None:
            self.val_sequences = []
        elif isinstance(val_sequences, str):
            self.val_sequences = [val_sequences]
        elif isinstance(val_sequences, list | ListConfig):
            self.val_sequences = (
                # pyrefly: ignore [bad-assignment]
                OmegaConf.to_container(val_sequences) if isinstance(val_sequences, ListConfig) else val_sequences
            )
        else:
            raise TypeError("Unrecognized type for val_sequences")
        if test_sequences is None:
            self.test_sequences = []
        elif isinstance(test_sequences, str):
            self.test_sequences = [test_sequences]
        elif isinstance(test_sequences, list | ListConfig):
            self.test_sequences = (
                # pyrefly: ignore [bad-assignment]
                OmegaConf.to_container(test_sequences) if isinstance(test_sequences, ListConfig) else test_sequences
            )
        else:
            raise TypeError("Unrecognized type for test_sequences")

        self.precomputed_std = precomputed_std
        self.num_aa_min = num_aa_min
        self.num_aa_max = num_aa_max
        self.system_cond_ids = system_cond_ids or []

        # Construct derived paths
        self.pdb_dir = Path(data_dir) / "pdbs"
        self.wds_cache_dir = Path(data_dir) / self.WDS_REPO_PATH
        self.evaluation_data_path = Path(data_dir) / "trajectories_subsampled"
        self.val_data_path = self.evaluation_data_path / "val"
        self.test_data_path = self.evaluation_data_path / "test"
        self.preproc_cache_dir = Path(data_dir) / "preprocessing_cache"
        self.pdb_dict_pkl_path = self.preproc_cache_dir / "pdb_dict.pkl"
        self.topology_dict_pkl_path = self.preproc_cache_dir / "topology_dict.pkl"
        self.encodings_dict_pkl_path = self.preproc_cache_dir / "encodings_dict.pkl"
        self.permutations_dict_pkl_path = self.preproc_cache_dir / "permutations_dict.pkl"

        # Precomputed std for standardization
        self.std = torch.tensor(self.precomputed_std)

    def prepare_data(self) -> None:
        """Download PDBs and evaluation data, then cache preprocessing dicts.

        Lightning ensures this is called only within a single process on CPU,
        so downloading logic is safe here. In multi-node training, execution
        depends on ``self.prepare_data_per_node()``.

        Do not use it to assign state (``self.x = y``).
        """
        self.wds_cache_dir.mkdir(parents=True, exist_ok=True)
        self.preproc_cache_dir.mkdir(parents=True, exist_ok=True)

        download_and_extract_pdb_tarfiles(self.data_dir)
        download_evaluation_data(self.data_dir)

        # Build or load cached preprocessing dicts (each function handles caching internally)
        pdb_paths = [str(p) for p in self.pdb_dir.glob("*/*.pdb")]
        # pyrefly: ignore [bad-argument-type]
        pdb_dict = prepare_and_cache_pdb_dict(pdb_paths, self.pdb_dict_pkl_path, delimiter=".")
        # pyrefly: ignore [bad-argument-type]
        topology_dict = prepare_and_cache_topology_dict(pdb_dict, self.topology_dict_pkl_path)

        if "encodings" in self.system_cond_ids:
            # pyrefly: ignore [bad-argument-type]
            prepare_and_cache_encodings_dict(topology_dict, self.encodings_dict_pkl_path)
        if "permutations" in self.system_cond_ids:
            # pyrefly: ignore [bad-argument-type]
            prepare_and_cache_permutations_dict(topology_dict, self.permutations_dict_pkl_path)

    def setup(self, stage: str | None = None) -> None:
        """Load cached preprocessing dicts and build the training dataset.

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

        if stage != "fit":
            return

        # Both WebDataset and buffer training require knowing the number of batches in advance
        # pyrefly: ignore [missing-attribute]
        assert isinstance(self.trainer.limit_train_batches, int), (
            "trainer.limit_train_batches must be set to an integer for wds and buffer training"
        )

        # Load cached data preprocessing dicts
        with self.pdb_dict_pkl_path.open("rb") as f:
            self.pdb_dict = pickle.load(f)  # noqa: S301
        logger.info(f"Loaded pdb dict from {self.pdb_dict_pkl_path}")
        with self.topology_dict_pkl_path.open("rb") as f:
            self.topology_dict = pickle.load(f)  # noqa: S301
        logger.info(f"Loaded topology dict from {self.topology_dict_pkl_path}")

        self.encodings_dict = None
        if "encodings" in self.system_cond_ids:
            with self.encodings_dict_pkl_path.open("rb") as f:
                self.encodings_dict = pickle.load(f)  # noqa: S301
            logger.info(f"Loaded encodings dict from {self.encodings_dict_pkl_path}")

        self.permutations_dict = None
        if "permutations" in self.system_cond_ids:
            with self.permutations_dict_pkl_path.open("rb") as f:
                self.permutations_dict = pickle.load(f)  # noqa: S301
            logger.info(f"Loaded permutations dict from {self.permutations_dict_pkl_path}")

        # Build transformations pipeline
        base_transform_list = [
            # pyrefly: ignore [bad-argument-type]
            StandardizeTransform(self.std),
            Random3DRotationTransform(),
        ]
        if self.com_augmentation:
            # pyrefly: ignore [bad-argument-type]
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
                # pyrefly: ignore [bad-argument-type]
                train_transform_list.append(AddEncodingsTransform(self.encodings_dict))
            if self.permutations_dict is not None:
                # pyrefly: ignore [bad-argument-type]
                train_transform_list.append(AddPermutationsTransform(self.permutations_dict))
            # pyrefly: ignore [bad-argument-type]
            train_transform_list.append(PaddingTransform(self.num_atoms))
            train_transforms = torchvision.transforms.Compose(train_transform_list)

            tar_urls = self._resolve_tar_urls()
            # pyrefly: ignore [bad-assignment]
            self.data_train = build_webdataset(
                tar_urls=tar_urls,
                # pyrefly: ignore [bad-argument-type]
                cache_dir=self.wds_cache_dir,
                num_aa_min=self.num_aa_min,
                num_aa_max=self.num_aa_max,
                transform=train_transforms,
            )

    # pyrefly: ignore [bad-function-definition]
    def prepare_eval(self, sequence: str, stage: str | None = None) -> EvalContext:
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
        assert stage == ("val" if is_val else "test"), (
            f"Stage '{stage}' does not match sequence '{sequence}' (found in {'val' if is_val else 'test'})."
        )

        data_path = self.val_data_path if is_val else self.test_data_path
        subsampled_trajectory_npz = np.load(
            data_path / f"{len(sequence)}AA" / f"{sequence}_subsampled.npz",
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
            permutations = (
                get_permutations_dict({sequence: topology})[sequence]
                if "permutations" in self.system_cond_ids
                else None
            )
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
        local_tars = sorted(str(p) for p in self.wds_cache_dir.glob("*.tar"))
        if len(local_tars) >= self.NUM_TARFILES:
            logger.info(f"Found {len(local_tars)} cached tar files in {self.wds_cache_dir}, skipping HuggingFace.")
            return local_tars
        if local_tars:
            logger.info(
                f"Found {len(local_tars)} cached tar files but expected {self.NUM_TARFILES}. "
                "Falling back to HuggingFace."
            )
        return None

    def _resolve_hf_tar_urls(self) -> list[str]:
        hf_url_prefix = f"https://huggingface.co/datasets/{self.HF_REPO_ID}/resolve/main/{self.WDS_REPO_PATH}"
        return [f"{hf_url_prefix}/{i:04d}.tar" for i in range(self.NUM_TARFILES)]

    def _resolve_tar_urls(self) -> list[str]:
        """Resolve tar file URLs, preferring local cache over HuggingFace."""
        local_tars = self._check_cache_for_tarfiles()
        if local_tars is not None:
            return local_tars
        urls = self._resolve_hf_tar_urls()
        logger.info(f"Using WebDataset cache dir: {self.wds_cache_dir}")
        return urls

    def _load_pdb(self, sequence: str, stage: str) -> Any:
        """Load PDB file for a sequence directly from disk.

        Args:
            sequence: Peptide sequence identifier.
            stage: Dataset split ("val" or "test"), used to locate the PDB subdirectory.
        """
        pdb_path = self.pdb_dir / stage / f"{sequence}.pdb"
        if not pdb_path.exists():
            raise FileNotFoundError(f"No PDB file found at {pdb_path}")
        return openmm.app.PDBFile(str(pdb_path))

    def _setup_potential(self, pdb: Any) -> OpenMMEnergy:
        """Set up an OpenMM potential energy function from a PDB object."""
        forcefield = openmm.app.ForceField("amber14-all.xml", "implicit/obc1.xml")
        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=openmm.app.CutoffNonPeriodic,
            # pyrefly: ignore [missing-attribute]
            nonbondedCutoff=2.0 * openmm.unit.nanometer,
            constraints=None,
        )
        # pyrefly: ignore [missing-attribute]
        integrator = openmm.LangevinMiddleIntegrator(
            # pyrefly: ignore [unsupported-operation]
            310 * openmm.unit.kelvin,
            # pyrefly: ignore [missing-attribute]
            0.3 / openmm.unit.picosecond,
            # pyrefly: ignore [missing-attribute]
            1.0 * openmm.unit.femtosecond,
        )
        platform_name = "CUDA" if torch.cuda.is_available() else "CPU"
        device_index = torch.cuda.current_device() if torch.cuda.is_available() else None
        return OpenMMEnergy(system, integrator, platform_name=platform_name, device_index=device_index)
