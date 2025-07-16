#!/bin/bash
python src/train.py -m launcher=mila_l40 \
experiment=training/single_system/tarflow_al2 \
model.net.in_channels=1 \
logger=wandb \
seed=0,1,2