#!/bin/bash
python src/train.py -m launcher=mila_l40 \
experiment=training/single_system/ecnf++_al2 \
tags=[sgb,al2,ecnf++_eval_v1] \
logger=wandb \
train=False \
ckpt_path="/network/scratch/t/tanc/sbg_final/ecnf++_al2_0.ckpt"