import logging

import torch
from lightning import Callback

from src.models.neural_networks.ema import EMA
from src.models.samplers.base_sampler_class import BaseSampler

logger = logging.getLogger(__name__)


class SelfImprovementCallback(Callback):
    """Lightning callback that generates self-improvement samples at the start of each training epoch.

    Owns its own sampler instance. On train_epoch_start:
    1. Swaps to EMA weights if applicable
    2. Generates samples via the sampler
    3. Adds resampled samples to the replay buffer
    """

    def __init__(self, sampler: BaseSampler):
        super().__init__()
        self.sampler = sampler

    def on_train_epoch_start(self, trainer, pl_module):
        datamodule = trainer.datamodule
        assert datamodule.test_sequences is not None, "Eval sequence name should be set"
        assert len(datamodule.test_sequences) == 1, "Can only self-refine on 1 test sequence at a time."
        assert datamodule.buffer is not None, "Need to have buffer instantiated in datamodule for self-consumption"

        logger.info(f"Generating {self.sampler.num_samples} samples for self-consumption")
        pl_module.net.eval()

        eval_ctx = datamodule.prepare_eval(datamodule.test_sequences[0], stage="test")

        use_ema = isinstance(pl_module.net, EMA) and pl_module.hparams.ema_decay > 0
        if use_ema:
            pl_module.net.backup()
            pl_module.net.copy_to_model()

        with torch.no_grad():
            samples_dict = self.sampler.sample(pl_module, eval_ctx.proposal_cond, eval_ctx.target_energy_fn)

        if use_ema:
            pl_module.net.restore_to_model()

        samples = samples_dict["resampled"].samples  # already unnormalized

        pl_module.net.train()

        datamodule.data_train.buffer.add(samples, datamodule.test_sequences[0])
        datamodule.save_buffer()
