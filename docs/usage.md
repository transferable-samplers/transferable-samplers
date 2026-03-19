# Usage

> If you haven't already, follow the [Quickstart](README.md#quickstart) to install the codebase.

## Datasets

Both ManyPeptidesMD and the single-peptide datasets are hosted on Hugging Face.
- [ManyPeptidesMD](https://huggingface.co/datasets/transferable-samplers/many-peptides-md)
- [Single systems](https://huggingface.co/datasets/transferable-samplers/sequential-boltzmann-generators-data)

In both cases, the codebase is set up to automatically download the necessary data for training and evaluation. In the case of ManyPeptidesMD the training webdataset will by default be streamed and cached.

## Model Weights

The pretrained model weights used in our works are provided [here](https://huggingface.co/transferable-samplers/model-weights).

## Training

The codebase builds on the [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template). Experiments are organized into configuration files under `configs/experiment/`.

Train a TarFlow on the single-system Ace-A-Nme dataset:

```bash
uv run python -m transferable_samplers.train experiment=single_system/train/tarflow_Ace-A-Nme
```

Train Prose on the ManyPeptidesMD dataset:

```bash
uv run python -m transferable_samplers.train experiment=transferable/train/prose_up_to_8aa
```

> **Note:**  Pass `trainer=ddp` for distributed data parallel training.

### Resuming

Pass `resume_ckpt_path` to resume a full Lightning checkpoint (model weights, optimizer state, scheduler, epoch counter). If the file does not yet exist, it is silently ignored and training starts from scratch, making it safe to set `resume_ckpt_path` upfront for preemptible jobs:

```bash
uv run python -m transferable_samplers.train experiment=transferable/train/prose_up_to_8aa resume_ckpt_path=/path/to/last.ckpt
```

### Finetuning

To finetune a pretrained model, use a regular training experiment config and pass `init_ckpt_path` or `init_hf_state_dict_path` to initialise from existing weights. This loads only the model weights — optimizer state and scheduler are reset and training proceeds from epoch 0:

```bash
# Initialise from a local checkpoint
uv run python -m transferable_samplers.train experiment=transferable/train/prose_up_to_8aa init_ckpt_path=/path/to/checkpoint.ckpt

# Initialise from HuggingFace weights
uv run python -m transferable_samplers.train experiment=transferable/train/prose_up_to_8aa init_hf_state_dict_path=transferable/prose_up_to_8aa.pth
```

`init_ckpt_path` and `init_hf_state_dict_path` are mutually exclusive.

> `resume_ckpt_path` takes priority over any `init_*` arguments. If the resume checkpoint exists, init weights are ignored.

## Evaluation

### Sampling

Evaluation configs set `hf_state_dict_path` to automatically download pretrained weights from Hugging Face. To use a locally trained checkpoint instead, pass `ckpt_path` — these two options are mutually exclusive:

```bash
# Default: downloads pretrained weights from HuggingFace
uv run python -m transferable_samplers.eval experiment=transferable/eval/prose_up_to_8aa_snis

# Override with a local checkpoint (.ckpt)
uv run python -m transferable_samplers.eval experiment=transferable/eval/prose_up_to_8aa_snis ckpt_path=/path/to/checkpoint.ckpt
```

The `val` and `test` flags control which split(s) are evaluated. By default `test=true` and `val=false`.

```bash
# Run on test split only (default)
uv run python -m transferable_samplers.eval experiment=transferable/eval/prose_up_to_8aa_snis

# Run on validation split only
uv run python -m transferable_samplers.eval experiment=transferable/eval/prose_up_to_8aa_snis val=true test=false

```

The data configs define lists of `val_sequences` and `test_sequences` which are evaluated in turn. To evaluate on a specific subset of systems, override the sequence list:

```bash
uv run python -m transferable_samplers.eval experiment=transferable/eval/prose_up_to_8aa_snis \
  data.test_sequences=[AA]
```

In practice it is often more convenient to run each evaluation separately and aggregate the per-system metrics afterwards, rather than running sequentially as a single process.

> **Note:**  All samplers support multiple GPUs - pass `trainer=ddp` for distributed sampling.

### Self-Improving Sampling

Self-improvement finetunes a pretrained model using SNIS-reweighted samples from its own proposal distribution. It uses the same `init_*` arguments as regular finetuning (see above), but with a dedicated experiment config:

```bash
# Default: initialises from pretrained weights on HuggingFace
uv run python -m transferable_samplers.train experiment=transferable/fine-tune/prose_up_to_8aa_self_improve

# Override with a local checkpoint (.ckpt)
uv run python -m transferable_samplers.train experiment=transferable/fine-tune/prose_up_to_8aa_self_improve init_ckpt_path=/path/to/checkpoint.ckpt
```

To resume a self-improvement run, pass `resume_ckpt_path` as with regular training.
