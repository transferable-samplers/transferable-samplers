import os
from pathlib import Path

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from src.eval import eval

@pytest.mark.slow
def test_eval(tmp_path: Path, cfg_eval: DictConfig) -> None:
    """Tests training and evaluation by training for 1 epoch with `train.py` then evaluating with
    `eval.py`.

    :param tmp_path: The temporary logging path.
    :param cfg_train: A DictConfig containing a valid training configuration.
    :param cfg_eval: A DictConfig containing a valid evaluation configuration.
    """
    assert str(tmp_path) == cfg_eval.paths.output_dir

    test_metric_dict, _ = eval(cfg_eval)

    # assert test_metric_dict["test/acc"] > 0.0
    # assert abs(train_metric_dict["test/acc"].item() - test_metric_dict["test/acc"].item()) < 0.001


