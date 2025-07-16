#!/bin/bash
python src/train.py -m launcher=mila_l40 \
experiment=training/single_system/ecnf++_al3 \
tags=[sgb,al3,ecnf++_eval_v1] \
logger=wandb \
train=False \
ckpt_path="/network/scratch/t/tanc/sbg_final/ecnf++_al3_0.ckpt","/network/scratch/t/tanc/sbg_final/ecnf++_al3_1.ckpt","/network/scratch/t/tanc/sbg_final/ecnf++_al3_2.ckpt"