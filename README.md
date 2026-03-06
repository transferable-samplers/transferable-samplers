<h2 align="center">
    <p>Transferable Sampling of Molecular Systems</p>
</h2>

## Welcome to Transferable Samplers!

Transferable samplers is a **research codebase** for sampling molecular systems. Our focus is methods that leverage chemical encodings to _transfer_ information from large pretraining datasets to previously unseen systems at inference time!

The codebase has been designed to be edited and hacked, to rapidly implement and evaluate novel methods.

**Transferable Samplers** is the official codebase for the following works:

**Amortized Sampling with Transferable Normalizing Flows** [*NeurIPS 2025*](https://arxiv.org/abs/2508.18175)

**Scalable Equilibrium Sampling with Sequential Boltzmann Generators** [*ICML 2025*](https://icml.cc/virtual/2025/poster/45137)

We additionally include baseline implementations for:

**Transferable Boltzmann Generators** [*NeurIPS 2024*](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5035a409f5798e188079e236f437e522-Abstract-Conference.html)

More to come soon!

## Installation

```bash
# Create and activate a virtual environment
uv venv .venv --python=3.11
source .venv/bin/activate

# Install in editable mode
uv pip install -e .

# Install runtime dependencies
uv pip install -r requirements.txt
```

You must also populate `.env.example` and save as `.env` (sets `SCRATCH_DIR`)

**Optional:** Flash Attention (for TarFlow-based methods) must be installed separately. See the comment at the bottom of `requirements.txt`.

## Usage

### Datasets

Both ManyPeptidesMD and the single-peptide datasets are hosted on Hugging Face.
- [ManyPeptidesMD](https://huggingface.co/datasets/transferable-samplers/many-peptides-md)
- [Single systems](https://huggingface.co/datasets/transferable-samplers/sequential-boltzmann-generators-data)

In both cases, the codebase is set up to automatically download the necessary data for training and evaluation. In the case of ManyPeptidesMD the training webdataset will by default be streamed and cached.

### Model weights

The pretrained model weights used in our works are provided [here](https://huggingface.co/transferable-samplers/model-weights).

### Training

The codebase builds on the [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template). Experiments are organized into configuration files under `configs/experiment/`.

Train a TarFlow on the single-system Ace-A-Nme dataset:

```bash
uv run python -m transferable_samplers.train experiment=single_system/train/tarflow_Ace-A-Nme
```

Train Prose on the ManyPeptidesMD dataset:

```bash
uv run python -m transferable_samplers.train experiment=transferable/train/prose_up_to_8aa
```

### Evaluation (Sampling)

The evaluation experiment configs will default to downloading and using the pretrained model weights provided [here](https://huggingface.co/transferable-samplers/model-weights).

```bash
uv run python -m transferable_samplers.eval experiment=transferable/eval/prose_up_to_8aa_snis
```

To use a locally generated checkpoint, pass `ckpt_path=<path>` to override the remote weights.

### Self-improvement (Fine-tuning)

Self-improvement training loads a pretrained model and fine-tunes with samples from the model's own proposal distribution:

```bash
uv run python -m transferable_samplers.train experiment=transferable/fine-tune/prose_up_to_8aa_self_improve
```

## Acknowledgements

We would like to thank Hugging Face for hosting our large ManyPeptidesMD dataset!

## License

The core of this repository is licensed under the MIT License (see [LICENSE](./LICENSE)).
Some files include adaptation of third-party code under other licenses (Apple, NVIDIA, Klein & Noé).
In some cases, these third-party licenses are **non-commercial**.
See [NOTICE](./NOTICE) for details.
