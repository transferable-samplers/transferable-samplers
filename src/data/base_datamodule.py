from abc import ABC, abstractmethod
from typing import Any, Optional

import webdataset as wds
from lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset

from src.utils.dataclasses import EvalContext


class BaseDataModule(LightningDataModule, ABC):
    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        persistent_workers: bool = False,
        pin_memory: bool = False,
    ) -> None:
        """Initialize a `BaseDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param train_val_test_split: The train, validation and test split. Defaults to `(55_000, 5_000, 10_000)`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory

    @abstractmethod
    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        ...

    @abstractmethod
    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        ...

    @abstractmethod
    def prepare_eval(self, sequence: str, stage: str, num_eval_samples: int) -> EvalContext:
        """Prepare evaluation data and energy function for a given peptide sequence.

        Args:
            sequence: Peptide sequence identifier to prepare evaluation data for.
            stage: Dataset split ("val" or "test"). Used to select data path.

        Returns:
            EvalContext with all components required for evaluation.
        """
        ...

    def _validate_and_set_batch_size(self) -> None:
        """Validate batch size is divisible by world size and set per-device batch size."""
        if self.batch_size % self.trainer.world_size != 0:
            raise RuntimeError(
                f"Batch size ({self.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
            )
        self.batch_size_per_device = self.batch_size // self.trainer.world_size

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        is_iterable = isinstance(self.data_train, IterableDataset)

        if is_iterable:
            data_loader = wds.WebLoader(
                self.data_train, batch_size=self.batch_size_per_device, num_workers=self.num_workers
            )

            # Define epoch length (can be overridden by Lightning's `limit_train_batches`)
            data_loader = data_loader.with_epoch(10_000)

            return data_loader

        else:
            return DataLoader(
                dataset=self.data_train,
                batch_size=self.batch_size_per_device,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=True,
                persistent_workers=self.persistent_workers,
            )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        NOTE: these only exist for Lightning compatibility. All evaluation is handled by custom callbacks.
        """
        world_size = self.trainer.world_size if self.trainer is not None else 1
        return DataLoader(dataset=self.data_val, batch_size=world_size)

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        NOTE: these only exist for Lightning compatibility. All evaluation is handled by custom callbacks.
        """
        world_size = self.trainer.world_size if self.trainer is not None else 1
        return DataLoader(dataset=self.data_test, batch_size=world_size)
