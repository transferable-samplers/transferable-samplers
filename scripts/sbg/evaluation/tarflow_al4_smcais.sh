#!/bin/bash
python src/train.py -m launcher=mila_48gb \
experiment=evaluation/single_system/tarflow_al4_fk \
tags=[sbg,tarflow,al4,smcais_v6] \
model.sampling_config.num_test_proposal_samples=10_000,100_000 \
model.sampling_config.use_com_adjustment=1 \
model.sampling_config.clip_reweighting_logits=0.002 \
model.smc_sampler.input_energy_cutoff=50.0 \
model.smc_sampler.num_timesteps=100 \
model.smc_sampler.langevin_eps=1e-8 \
model.smc_sampler.ess_threshold=0.99,0.9 \
model.smc_sampler.systematic_resampling=True \
model.smc_sampler.batch_size=512 \
ckpt_path="/home/mila/t/tanc/scratch/self-consume-bg/logs/train/multiruns/2025-05-30_04-42-37/0/checkpoints/epoch_499_resampled_energy_w2.ckpt","/home/mila/t/tanc/scratch/self-consume-bg/logs/train/multiruns/2025-05-30_04-42-37/1/checkpoints/epoch_449_resampled_energy_w2.ckpt","/home/mila/t/tanc/scratch/self-consume-bg/logs/train/multiruns/2025-05-30_04-42-37/2/checkpoints/epoch_799_resampled_energy_w2.ckpt" 