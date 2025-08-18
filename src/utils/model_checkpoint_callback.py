import torch
from lightning.pytorch.callbacks import ModelCheckpoint


class FullAndWeightsModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _save_checkpoint(self, trainer, filepath):
        # 1) Let the normal ModelCheckpoint save the full checkpoint
        super()._save_checkpoint(trainer, filepath)

        # 2) Save weights-only version with a modified filename
        weights_path = filepath.replace(".ckpt", "_weights_only.ckpt")
        torch.save(trainer.lightning_module.state_dict(), weights_path)
