# Ensemble

## Dev Setup

```
pip install ruff
pre-commit install
```

## Install
```
conda create -n ensemble python=3.11
conda activate ensemble
pip install -r requirements.txt
```

## Train
```
python src/train.py experiment=training/single_system/tarflow_up_to_8aa trainer=gpu
```

## Sampling
```
python src/eval.py ckpt_path=${CHECKPOINT_PATH} experiment=evaluation/single_system/tarflow_up_to_8aa

```
