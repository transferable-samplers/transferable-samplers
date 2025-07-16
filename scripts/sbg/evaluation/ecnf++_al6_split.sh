#!/bin/bash
python src/train.py -m launcher=mila_l40 \
experiment=evaluation/single_system/ecnf++_al6_split \
tags=[sgb,al6,ecnf++_al6_split_eval_v1] \
logger=wandb \
model.sampling_config.num_test_proposal_samples=500 \
+model.sampling_config.subset_idx="range(0, 20)" \
ckpt_path="/network/scratch/t/tanc/sbg_final/ecnf++_al6_0.ckpt","/network/scratch/t/tanc/sbg_final/ecnf++_al6_1.ckpt","/network/scratch/t/tanc/sbg_final/ecnf++_al6_2.ckpt"