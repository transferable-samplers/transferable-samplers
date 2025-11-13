from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

RELATIVE_CONFIG_PATH = "../../configs"  # relative to this utils.py file


def compose_config(config_name: str, overrides=None):
    GlobalHydra.instance().clear()
    with initialize(version_base="1.3", config_path=RELATIVE_CONFIG_PATH):
        cfg = compose(config_name=config_name, overrides=overrides)
    return cfg


def extract_test_sequence(cfg):
    seq = getattr(cfg.data, "sequence", None) or cfg.data.test_sequences
    return seq[0] if isinstance(seq, list) else seq
