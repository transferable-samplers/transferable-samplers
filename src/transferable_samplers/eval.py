import random
import time
from typing import Any

from dotenv import load_dotenv

load_dotenv(override=True)

import hydra
import lightning
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from transferable_samplers.utils.hydra_utils import extras, get_metric_value, task_wrapper
from transferable_samplers.utils.init_resume_utils import resolve_init
from transferable_samplers.utils.instantiators import instantiate_callbacks, instantiate_loggers
from transferable_samplers.utils.pylogger import RankedLogger
from transferable_samplers.utils.wandb_utils import log_hyperparameters

# We had issues with invertbility of TarFlow without the following settings.
# Didn't notice any walltime difference for TarFlow or ECNF, but is worth
# benchmarking for any further implemented models.
torch.set_float32_matmul_precision("highest")  # must be at least high
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
logger = RankedLogger(__name__, rank_zero_only=False)
logger.info(
    "Some numerical settings applied for TarFlow invertibility. No slowdown was "
    "observed for ECNF but other neural networks may be slower than expected."
)


@task_wrapper
def eval(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Evaluate a model on validation and/or test set.

    Wrapped in @task_wrapper for failure handling during multiruns.

    Model weights are loaded via ``cfg.ckpt_path`` (Lightning checkpoint) or
    ``cfg.hf_state_dict_path`` (Hugging Face Hub). At least one of ``cfg.val``
    or ``cfg.test`` must be enabled.

    Args:
        cfg: Hydra DictConfig. Key fields used here:
            - ``cfg.ckpt_path``: Path to a checkpoint for weight initialisation.
            - ``cfg.hf_state_dict_path``: HF Hub path for weight initialisation.
            - ``cfg.seed``: Random seed for reproducibility.
            - ``cfg.val``: Whether to run validation.
            - ``cfg.test``: Whether to run testing.

    Returns:
        A tuple of (merged val/test metrics dict, object dict with all instantiated objects).
    """
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        lightning.seed_everything(cfg.seed, workers=True)

    if cfg.get("torch_num_threads"):
        torch.set_num_threads(cfg.torch_num_threads)

    logger.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    logger.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    logger.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    logger.info("Instantiating loggers...")
    loggers: list[Logger] = instantiate_loggers(cfg.get("logger"))

    logger.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=loggers
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": loggers,
        "trainer": trainer,
    }

    if loggers:
        logger.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    init_state_dict = resolve_init(
        init_ckpt_path=cfg.get("ckpt_path"),
        init_hf_state_dict_path=cfg.get("hf_state_dict_path"),
        scratch_dir=cfg.paths.scratch_dir,
    )

    if init_state_dict is not None:
        model.load_state_dict(init_state_dict)

    assert cfg.get("val", False) or cfg.get("test", False), "At least one of validation or test must be enabled!"

    if cfg.get("val"):
        logger.info("Starting validation!")
        trainer.validate(model=model, datamodule=datamodule)
        val_metrics = trainer.callback_metrics
    else:
        val_metrics = {}

    if cfg.get("test"):
        logger.info("Starting testing!")
        trainer.test(model=model, datamodule=datamodule)
        test_metrics = trainer.callback_metrics
    else:
        test_metrics = {}

    # merge train and test metrics
    metric_dict = {**val_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../../configs", config_name="eval.yaml")
def main(cfg: DictConfig) -> float | None:
    """Main entry point for evaluation.

    Args:
        cfg: DictConfig configuration composed by Hydra.

    Returns:
        Optimized metric value, if configured.
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
        logger.info(f"Sleeping for {sleep_time:.2f} seconds to avoid wandb rate limitations.")
        time.sleep(sleep_time)

    # train the model
    # pyrefly: ignore [bad-argument-type]
    metric_dict, _ = eval(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
