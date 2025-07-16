#!/bin/bash
python src/train.py -m launcher=mila_l40 \
experiment=evaluation/single_system/tarflow_chignolin_smc \
tags=[sbg,tarflow,chignolin,smc] \
model.sampling_config.num_test_proposal_samples=10_000 \
model.sampling_config.use_com_adjustment=1 \
model.sampling_config.clip_reweighting_logits=0.002 \
model.smc_sampler.input_energy_cutoff=-300 \
model.smc_sampler.num_timesteps=100 \
model.smc_sampler.langevin_eps=1e-8,1e-9 \
model.smc_sampler.ess_threshold=0.9,0.99 \
model.smc_sampler.systematic_resampling=True \
model.smc_sampler.batch_size=128 \
ckpt_path="/home/mila/t/tanc/scratch/sbg_final/tarflow_chignolin_0.ckpt"