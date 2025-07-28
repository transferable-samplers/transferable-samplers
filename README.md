# Transferable Samplers!

Welcome to the official codebase for "Scalable Equilibrium Sampling with Sequential Boltzmann Generators" (ICML 2025)[https://icml.cc/virtual/2025/poster/45137]

## Install

```
conda create -n transferable-samplers python=3.11
conda activate transferable-samplers
pip install -r requirements.txt
```

## Train
```
python src/train.py experiment=training/single_system/tarflow_al2 trainer=gpu
```

## Sampling
```
python src/eval.py ckpt_path=${CHECKPOINT_PATH} experiment=evaluation/tarflow_al2_smc_cont trainer=gpu

```
