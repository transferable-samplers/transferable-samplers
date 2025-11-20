import logging
from typing import Any, Callable, Optional

import webdataset as wds
from lightning import LightningDataModule
from torch.utils.data import DataLoader, IterableDataset

from src.data.preprocessing.tica import TicaModel
from src.utils.data_types import SamplesData


class BaseDataModule(LightningDataModule):
    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
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

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        raise NotImplementedError

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        raise NotImplementedError

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        is_iterable = isinstance(self.data_train, IterableDataset)

        if is_iterable:
            data_loader = wds.WebLoader(
                self.data_train, batch_size=self.batch_size_per_device, num_workers=self.hparams.num_workers
            )

            # Define dummy epoch length (can be overridden by Lightning's `limit_train_batches`)
            data_loader = data_loader.with_epoch(100_000)

            return data_loader

        else:
            persistent_workers_flag = True if self.hparams.num_workers > 0 else False
            num_workers = self.hparams.num_workers
            if hasattr(self.data_train, "buffer"):
                persistent_workers_flag = False
                num_workers = 0

            return DataLoader(
                dataset=self.data_train,
                batch_size=self.batch_size_per_device,
                num_workers=num_workers,
                pin_memory=self.hparams.pin_memory,
                shuffle=True,
                persistent_workers=persistent_workers_flag,
            )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,  # i shuffle in case of trainer.limit_val_batches
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,  # shuffle in case of trainer.limit_test_batches
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )

    def normalize(self, x):
        """
        Normalize trajectory samples using stored standard deviation.

        Subtracts the center of mass for each sample and divides by the scalar
        standard deviation computed previously.

        Args:
            x (torch.Tensor): Tensor of shape (num_samples, num_atoms, num_dimensions).

        Returns:
            torch.Tensor: Normalized tensor of the same shape.
        """
        assert self.std is not None, "Standard deviation should be computed first"
        assert self.std.numel() == 1, "Standard deviation should be scalar"
        assert len(x.shape) == 3, "Input should be 3D tensor"
        x = x - x.mean(dim=1, keepdim=True)
        x = x / self.std
        return x

    def unnormalize(self, x):
        """
        Undo normalization of trajectory samples using stored standard deviation.

        Multiplies normalized samples by the scalar standard deviation.

        Args:
            x (torch.Tensor): Normalized tensor of shape (num_samples, num_atoms, num_dimensions).

        Returns:
            torch.Tensor: Unnormalized tensor of the same shape.
        """
        assert self.std is not None, "Standard deviation should be computed first"
        assert self.std.numel() == 1, "Standard deviation should be scalar"
        assert len(x.shape) == 3, "Input should be 3D tensor"
        x = x * self.std.to(x)
        return x



if __name__ == "__main__":
    _ = BaseDataModule()
