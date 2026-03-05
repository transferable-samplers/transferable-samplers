from __future__ import annotations

import statistics as stats
from collections import defaultdict
from collections.abc import Callable
from typing import Any

import torch
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf

from transferable_samplers.utils.pylogger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=False)


def compute_mean_metrics(metrics: dict[str, Any], prefix: str = "val") -> dict[str, float]:
    """Aggregate per-sequence metrics by computing their mean.

    Args:
        metrics: Dict of metric values keyed by ``{prefix}/{sequence}/{metric_name}``.
        prefix: Metric key prefix to filter on (e.g. ``"val"``).

    Returns:
        Dict keyed by ``{prefix}/mean/{metric_name}`` with mean values.
    """
    mean_dict_list: defaultdict[str, list[float]] = defaultdict(list)

    for key, value in metrics.items():
        if key.startswith(prefix):
            parts = key.split("/")
            metric_name = "/".join(parts[2:])

            if isinstance(value, torch.Tensor):
                value = value.item()
            elif isinstance(value, int | float):
                value = float(value)

            mean_dict_list[f"{prefix}/mean/{metric_name}"].append(value)

    return {k: stats.mean(v) for k, v in mean_dict_list.items()}


def make_log_image_fn(trainer: Trainer) -> Callable[[Any, str | None, str], None]:
    """Return a safe image logger function.

    Logs only on global rank 0. Returns a no-op if no WandbLogger is present.

    Args:
        trainer: The Lightning trainer whose loggers are inspected.

    Returns:
        A callable ``(img, title, title_prefix) -> None`` that logs an image.
    """
    if not getattr(trainer, "is_global_zero", False):
        # pyrefly: ignore [bad-return]
        return lambda img, title=None, title_prefix="": None

    wandb_logger = None
    for lg in getattr(trainer, "loggers", []) or []:
        if isinstance(lg, WandbLogger):
            wandb_logger = lg
            break

    if wandb_logger is None:
        # pyrefly: ignore [bad-return]
        return lambda img, title=None, title_prefix="": None

    def log_image(img: Any, title: str | None = None, title_prefix: str = "") -> None:
        full_title = f"{title_prefix}/{title}" if title_prefix and title else (title or title_prefix)
        # pyrefly: ignore [missing-attribute]
        wandb_logger.log_image(full_title, [img])

    return log_image


@rank_zero_only
def log_hyperparameters(object_dict: dict[str, Any], resolve: bool = True) -> None:
    """Control which config parts are saved by Lightning loggers.

    Additionally saves the number of model parameters (total, trainable,
    non-trainable).

    Args:
        object_dict: A dictionary containing the following objects:
            - ``"cfg"``: A DictConfig object containing the main config.
            - ``"model"``: The Lightning model.
            - ``"trainer"``: The Lightning trainer.
        resolve: Whether to resolve OmegaConf references when converting the config.
    """
    hparams: dict[str, Any] = {}
    cfg: dict[str, Any] = OmegaConf.to_container(object_dict["cfg"], resolve=resolve)  # type: ignore[assignment]
    model = object_dict["model"]
    trainer = object_dict["trainer"]

    if not trainer.logger:
        logger.warning("Logger not found! Skipping hyperparameter logging...")
        return

    hparams["model"] = cfg["model"]

    # save number of model parameters
    hparams["model/params/total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params/trainable"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    hparams["model/params/non_trainable"] = sum(p.numel() for p in model.parameters() if not p.requires_grad)

    hparams["data"] = cfg["data"]
    hparams["trainer"] = cfg["trainer"]

    hparams["callbacks"] = cfg.get("callbacks")
    hparams["extras"] = cfg.get("extras")

    hparams["task_name"] = cfg.get("task_name")
    hparams["tags"] = cfg.get("tags")
    hparams["ckpt_path"] = cfg.get("ckpt_path")
    hparams["seed"] = cfg.get("seed")
    hparams["paths"] = cfg.get("paths")

    # send hparams to all loggers
    for lg in trainer.loggers:
        lg.log_hyperparams(hparams)
