#!/bin/bash
python src/train.py -m launcher=mila_l40 \
experiment=training/single_system/ecnf++_al6 \
model.optimizer.weight_decay=1e-4 \
logger=wandb \
seed=0,1,2