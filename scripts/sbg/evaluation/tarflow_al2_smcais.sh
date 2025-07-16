#!/bin/bash
python src/train.py -m launcher=mila_48gb \
experiment=evaluation/single_system/tarflow_al2_fk \
tags=[sbg,tarflow,al2,smcais_v6] \
model.sampling_config.num_test_proposal_samples=10_000,100_000 \
model.sampling_config.use_com_adjustment=1 \
model.sampling_config.clip_reweighting_logits=0.002 \
model.smc_sampler.input_energy_cutoff=10.0 \
model.smc_sampler.num_timesteps=100 \
model.smc_sampler.langevin_eps=1e-7 \
model.smc_sampler.ess_threshold=0.99,0.9 \
model.smc_sampler.systematic_resampling=True \
model.smc_sampler.batch_size=1024 \
ckpt_path="/home/mila/t/tanc/scratch/self-consume-bg/logs/train/multiruns/2025-05-30_04-43-26/0/checkpoints/epoch_849_resampled_energy_w2.ckpt","/home/mila/t/tanc/scratch/self-consume-bg/logs/train/multiruns/2025-05-30_04-43-26/1/checkpoints/epoch_349_resampled_energy_w2.ckpt","/home/mila/t/tanc/scratch/self-consume-bg/logs/train/multiruns/2025-05-30_04-43-26/2/checkpoints/epoch_399_resampled_energy_w2.ckpt"