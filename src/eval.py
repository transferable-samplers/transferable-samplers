# ruff: noqa: E402, I001

from typing import Any, Optional
import logging
import os


import time
import random
import hydra
import lightning as L
import rootutils
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig
from dotenv import load_dotenv

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
load_dotenv(override=True)

# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils.huggingface import download_weights
from src.utils.instantiators import instantiate_callbacks, instantiate_loggers
from src.utils.logging_utils import log_hyperparameters
from src.utils.pylogger import RankedLogger
from src.utils.utils import extras, get_metric_value, task_wrapper

# We had issues with invertbility of TarFlow without the following settings.
# Didn't notice any walltime difference for TarFlow or ECNF, but is worth
# benchmarking for any further implemented models.
torch.set_float32_matmul_precision("highest")  # must be at least high
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
logging.info(
    "Some numerical settings applied for TarFlow invertibility. No slowdown was "
    "observed for ECNF but other neural networks may be slower than expected.",
)
# TODO consolidate codebase logging into single library.
log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def eval(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    if cfg.get("torch_num_threads"):
        torch.set_num_threads(cfg.torch_num_threads)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model, datamodule=datamodule)

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    ckpt_path = cfg.get("ckpt_path")
    state_dict_hf_path = cfg.get("state_dict_hf_path")

    # Ensure exactly one of ckpt_path or state_dict_hf_path is provided (XOR)
    assert (ckpt_path is None) ^ (state_dict_hf_path is None), "You must provide one of ckpt_path or state_dict_hf_path"

    if state_dict_hf_path is not None:
        # Provided a remote state dict path
        log.info("Downloading weights from huggingface...")
        dst_dir = os.path.join(cfg.paths.scratch_dir, "model-weights")
        state_dict_path = download_weights(hf_filepath=state_dict_hf_path, destination_dir=dst_dir)

        # Directly load the state dict into model
        assert not cfg.model.ema_decay, (
            "Setting ema decay will cause silent errors in evaluation when using huggingface weights"
        )
        state_dict = torch.load(state_dict_path, map_location="cpu")
        model.load_state_dict(state_dict)

    assert cfg.get("val", False) or cfg.get("test", False), "At least one of validation or test must be enabled!"

    if cfg.get("val"):
        log.info("Starting validation!")
        trainer.validate(
            model=model,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
        )  # ckpt_path is None if using state_dict_hf_path
        val_metrics = trainer.callback_metrics
    else:
        val_metrics = {}

    if cfg.get("test"):
        log.info("Starting testing!")
        trainer.test(
            model=model,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
        )  # ckpt_path is None if using state_dict_hf_path
        test_metrics = trainer.callback_metrics
    else:
        test_metrics = {}

    # merge train and test metrics
    metric_dict = {**val_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: Optional[float] with optimized metric value.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    if "multiruns" in cfg.paths.output_dir:
        # We had an issue with multirun sweeps where > 100 jobs would start at the same time,
        # causing rate limiting issues with wandb. To avoid this, we add a random sleep time
        # before starting the training. This is a workaround to avoid hitting the rate limits.
        # It seems to be fine having many concurrent jobs, but not starting simultaneously.
        sleep_time = random.uniform(0, 60)  # noqa:S311
        log.info(f"Sleeping for {sleep_time:.2f} seconds to avoid wandb rate limitations.")
        time.sleep(sleep_time)

    # train the model
    metric_dict, _ = eval(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
