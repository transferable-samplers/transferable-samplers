import statistics as stats
from collections import defaultdict
from typing import Any, Callable, Optional

import torch
from lightning.pytorch.loggers import WandbLogger
from lightning_utilities.core.rank_zero import rank_zero_only
from omegaconf import OmegaConf

from src.utils.pylogger import RankedLogger

logger = RankedLogger(__name__, rank_zero_only=False)


def compute_mean_metrics(metrics: dict, prefix: str = "val") -> dict:
    """Aggregate metrics across all sequences by computing mean."""
    mean_dict_list = defaultdict(list)

    for key, value in metrics.items():
        if key.startswith(prefix):
            parts = key.split("/")
            metric_name = "/".join(parts[2:])

            if isinstance(value, torch.Tensor):
                value = value.item()
            elif isinstance(value, (int, float)):
                value = float(value)

            mean_dict_list[f"{prefix}/mean/{metric_name}"].append(value)

    return {k: stats.mean(v) for k, v in mean_dict_list.items()}


def make_log_image_fn(trainer) -> Callable[[Any, Optional[str]], None]:
    """Return a safe image logger function.

    - Logs only on global rank 0
    - If no WandbLogger is present, becomes a no-op
    """
    if not getattr(trainer, "is_global_zero", False):
        return lambda img, title=None, title_prefix="": None

    wandb_logger = None
    for lg in getattr(trainer, "loggers", []) or []:
        if isinstance(lg, WandbLogger):
            wandb_logger = lg
            break

    if wandb_logger is None:
        return lambda img, title=None, title_prefix="": None

    def log_image(img, title: Optional[str] = None, title_prefix: str = ""):
        full_title = f"{title_prefix}/{title}" if title_prefix and title else (title or title_prefix)
        wandb_logger.log_image(full_title, [img])

    return log_image


@rank_zero_only
def log_hyperparameters(object_dict: dict[str, Any], resolve: bool = True) -> None:
    """Controls which config parts are saved by Lightning loggers.

    Additionally saves:
        - Number of model parameters

    :param object_dict: A dictionary containing the following objects:
        - `"cfg"`: A DictConfig object containing the main config.
        - `"model"`: The Lightning model.
        - `"trainer"`: The Lightning trainer.
    """
    hparams = {}
    cfg = OmegaConf.to_container(object_dict["cfg"], resolve=resolve)
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
