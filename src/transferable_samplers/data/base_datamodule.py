from abc import ABC, abstractmethod
from typing import Any

import webdataset as wds
from lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset

from transferable_samplers.utils.dataclasses import EvalContext


class BaseDataModule(LightningDataModule, ABC):
    """Abstract base class for peptide conformation datamodules.

    Subclasses must implement:
        - ``prepare_data``: Download / preprocess data (single process, no state).
        - ``setup``: Load data and set ``self.data_train``, ``self.data_val``,
          ``self.data_test``.
        - ``prepare_eval``: Build an ``EvalContext`` for a given peptide sequence.

    Val/test dataloaders are placeholders — actual evaluation is handled by
    ``SamplingEvaluationCallback`` via ``prepare_eval``.

    Args:
        data_dir: Root data directory.
        num_dimensions: Spatial dimensions per atom (typically 3).
        num_atoms: Maximum number of atoms per molecule (used for padding).
        batch_size: Global batch size (split across devices).
        num_workers: DataLoader worker count.
        persistent_workers: Keep workers alive between epochs.
        pin_memory: Pin DataLoader memory for faster GPU transfer.
        com_augmentation: Apply Gaussian center-of-mass augmentation.
        num_eval_samples: Number of conformations subsampled for evaluation.
        train_from_buffer: If True, training data comes from a replay buffer
            populated by ``PopulateBufferCallback`` instead of a dataset.
    """

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
    ) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_dir = data_dir
        self.num_dimensions = num_dimensions
        self.num_atoms = num_atoms
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.com_augmentation = com_augmentation
        self.num_eval_samples = num_eval_samples
        self.train_from_buffer = train_from_buffer

    @abstractmethod
    def prepare_data(self) -> None:
        """Download and preprocess data if needed.

        Lightning ensures this is called only within a single process on CPU,
        so downloading logic is safe here. In multi-node training, execution
        depends on ``self.prepare_data_per_node()``.

        Do not use it to assign state (``self.x = y``).
        """
        ...

    @abstractmethod
    def setup(self, stage: str | None = None) -> None:
        """Load data and set ``self.data_train``, ``self.data_val``, ``self.data_test``.

        Called by Lightning before ``trainer.fit()``, ``trainer.validate()``,
        ``trainer.test()``, and ``trainer.predict()``. A barrier after
        ``prepare_data`` ensures all processes wait until data is ready.

        Args:
            stage: One of ``"fit"``, ``"validate"``, ``"test"``, or ``"predict"``.
        """
        ...

    @abstractmethod
    def prepare_eval(self, sequence: str, stage: str) -> EvalContext:
        """Build an EvalContext for a given peptide sequence.

        Args:
            sequence: Peptide sequence identifier.
            stage: Dataset split (``"val"`` or ``"test"``).

        Returns:
            EvalContext containing true conformations, target energy, topology,
            TICA model, and system conditioning (if applicable).
        """
        ...

    def _validate_and_set_batch_size(self) -> None:
        """Validate batch size is divisible by world size and set per-device batch size."""
        # pyrefly: ignore [missing-attribute]
        if self.batch_size % self.trainer.world_size != 0:
            raise RuntimeError(
                # pyrefly: ignore [missing-attribute]
                f"Batch size ({self.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
            )
        # pyrefly: ignore [missing-attribute]
        self.batch_size_per_device = self.batch_size // self.trainer.world_size

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader."""
        # pyrefly: ignore [missing-attribute]
        is_iterable = isinstance(self.data_train, IterableDataset)

        if is_iterable:
            data_loader = wds.WebLoader(
                # pyrefly: ignore [missing-attribute]
                self.data_train,
                batch_size=self.batch_size_per_device,
                num_workers=self.num_workers,
            )

            # Define epoch length (can be overridden by Lightning's `limit_train_batches`)
            data_loader = data_loader.with_epoch(10_000)

            # pyrefly: ignore [bad-return]
            return data_loader

        else:
            return DataLoader(
                # pyrefly: ignore [missing-attribute]
                dataset=self.data_train,
                batch_size=self.batch_size_per_device,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=True,
                persistent_workers=self.persistent_workers,
            )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return a placeholder validation dataloader (evaluation handled by callbacks)."""
        world_size = self.trainer.world_size if self.trainer is not None else 1
        # pyrefly: ignore [missing-attribute]
        return DataLoader(dataset=self.data_val, batch_size=world_size)

    def test_dataloader(self) -> DataLoader[Any]:
        """Return a placeholder test dataloader (evaluation handled by callbacks)."""
        world_size = self.trainer.world_size if self.trainer is not None else 1
        # pyrefly: ignore [missing-attribute]
        return DataLoader(dataset=self.data_test, batch_size=world_size)
