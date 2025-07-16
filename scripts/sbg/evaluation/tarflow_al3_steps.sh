#!/bin/bash
python src/train.py -m launcher=mila_rtx8000 \
experiment=evaluation/single_system/tarflow_al3_fk \
tags=[sbg,tarflow,al3,smc_steps_v1] \
model.sampling_config.num_test_proposal_samples=10_000 \
model.sampling_config.use_com_adjustment=1,0 \
model.sampling_config.clip_reweighting_logits=0.002,null \
model.smc_sampler.input_energy_cutoff=-120.0 \
model.smc_sampler.num_timesteps=10,20,50,100,200,500,1000 \
model.smc_sampler.langevin_eps=1e-7 \
model.smc_sampler.ess_threshold=0.5 \
model.smc_sampler.batch_size=1024 \
ckpt_path="/home/mila/t/tanc/scratch/self-consume-bg/logs/train/multiruns/2025-05-30_04-42-59/0/checkpoints/epoch_599_resampled_energy_w2.ckpt","/home/mila/t/tanc/scratch/self-consume-bg/logs/train/multiruns/2025-05-30_04-42-59/1/checkpoints/epoch_799_resampled_energy_w2.ckpt","/home/mila/t/tanc/scratch/self-consume-bg/logs/train/multiruns/2025-05-30_04-42-59/2/checkpoints/epoch_999_resampled_energy_w2.ckpt"