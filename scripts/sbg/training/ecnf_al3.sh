#!/bin/bash
python src/train.py -m launcher=mila_l40 \
experiment=training/single_system/ecnf_al3 \
logger=wandb \
seed=0,1,2