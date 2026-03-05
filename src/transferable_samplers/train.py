# ruff: noqa: S311
import logging
import random
import time
from typing import Any

import hydra
import lightning
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from transferable_samplers.utils.hydra_utils import extras, get_metric_value, task_wrapper
from transferable_samplers.utils.init_resume_utils import augment_state_dict_for_teacher, resolve_init_or_resume
from transferable_samplers.utils.instantiators import instantiate_callbacks, instantiate_loggers
from transferable_samplers.utils.pylogger import RankedLogger
from transferable_samplers.utils.wandb_utils import log_hyperparameters

# We had issues with invertbility of TarFlow without the following settings.
# Didn't notice any walltime difference for TarFlow or ECNF, but is worth
# benchmarking for any further implemented models.
torch.set_float32_matmul_precision("highest")  # must be at least high
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
logging.info(
    "Some numerical settings applied for TarFlow invertibility. No slowdown was "
    "observed for ECNF but other neural networks may be slower than expected."
)
logger = RankedLogger(__name__, rank_zero_only=False)


@task_wrapper
def train(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train the model.

    Wrapped in @task_wrapper for failure handling during multiruns.

    Supports two checkpoint modes (mutually exclusive):
        - **Resume**: set ``resume_ckpt_path`` to a full Lightning checkpoint
          path (optimizer state, scheduler, epoch, etc.). If the file exists
          on disk, training resumes from it. If the file does not yet exist
          (e.g. first run of a preemptible job), it is ignored and training
          starts from scratch or from init weights instead.
        - **Init**: pass ``init_ckpt_path`` or ``init_hf_state_dict_path`` to
          load only model weights (e.g. for fine-tuning). If the model has a
          teacher, the state dict is augmented automatically.

    Args:
        cfg: Hydra DictConfig. Key fields used here:
            - ``cfg.resume_ckpt_path``: Path to a Lightning checkpoint to resume
              from. If the file doesn't exist yet, training starts fresh
              (safe for preemptible jobs that point to a future checkpoint path).
            - ``cfg.init_ckpt_path``: Path to a checkpoint for weight initialisation only.
            - ``cfg.init_hf_state_dict_path``: HF Hub path for weight initialisation.
            - ``cfg.seed``: Random seed for reproducibility.

    Returns:
        A tuple of (training metrics dict, object dict with all instantiated objects).
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
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

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

    # This returns fit_ckpt_path if we are resuming from an existing checkpoint.
    # Otherwise, it returns None and optionally an init_state_dict if we are
    # initializing the model from a checkpoint (e.g. for fine-tuning).
    fit_ckpt_path, init_state_dict = resolve_init_or_resume(
        resume_ckpt_path=cfg.get("resume_ckpt_path"),
        init_ckpt_path=cfg.get("init_ckpt_path"),
        init_hf_state_dict_path=cfg.get("init_hf_state_dict_path"),
        scratch_dir=cfg.paths.scratch_dir,
    )

    if init_state_dict is not None:
        assert fit_ckpt_path is None, "fit_ckpt_path must be None when init_state_dict is provided."
        # If the model has a teacher (i.e. teacher regularization), we need to augment the state dict before loading.
        if hasattr(model, "teacher"):
            init_state_dict = augment_state_dict_for_teacher(init_state_dict)
        logger.info("Loading model state dict from init_state_dict")
        model.load_state_dict(init_state_dict)

    logger.info("Starting training!")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=fit_ckpt_path)

    train_metrics = trainer.callback_metrics

    return train_metrics, object_dict


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> float | None:
    """Main entry point for training.

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
    metric_dict, _ = train(cfg)

    # safely retrieve metric value for hydra-based hyperparameter optimization
    metric_value = get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))

    # return optimized metric
    return metric_value


if __name__ == "__main__":
    main()
