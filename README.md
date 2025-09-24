# Transferable Samplers

Welcome to the official codebase for the following works:

**Amortized Sampling with Transferable Normalizing Flows** *Preprint*

**Scalable Equilibrium Sampling with Sequential Boltzmann Generators** [*ICML 2025*](https://icml.cc/virtual/2025/poster/45137)

## Installation

```
conda create -n transferable-samplers python=3.11
conda activate transferable-samplers
pip install -r requirements.txt
```

You must also populate `.env.example` and save as `.env`

## Usage

### Datasets

Both ManyPeptidesMD and the single-system datasets are hosted on Hugging Face.
- [ManyPeptidesMD](https://huggingface.co/datasets/transferable-samplers/many-peptides-md)
- [Single systems](https://huggingface.co/datasets/transferable-samplers/sequential-boltzmann-generators-data)

In both cases the codebase is setup to automatically download the necessary data for training and evaluation. In the case of ManyPeptidesMD the training webdataset will by default be streamed and cached.

### Model weights

The pretrained model weights used in our works are provided [here](https://huggingface.co/transferable-samplers/model-weights).

### Training

The codebase builds on the [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template). Accordingly, the experiments are organized into experiment configuration files.

To train a TarFlow on the single system Ace-A-Nme dataset run:

```
python src/train.py experiment=training/single_system/tarflow_Ace-A-Nme
```

To train Prose on the ManyPeptidesMD dataset

```
python src/train.py experiment=training/transferable/prose_up_to_8aa
```

## Sampling

The sampling experiment configs will default to downloading and using the pretrained model weights provided [here](https://huggingface.co/transferable-samplers/model-weights).

```
python src/eval_only.py experiment=evaluation/transferable/prose_up_to_8aa
```

To use a locally generated checkpoint you may pass in the argument `ckpt_path` to override the remote weights usage.

## Acknowledgements

We greatly thank Hugging Face for hosting our large ManyPeptidesMD dataset!

## License

The core of this repository is licensed under the MIT License (see [LICENSE](./LICENSE)).  
Some files include adaptation of third-party code under other licenses (Apple, Meta, NVIDIA, Klein & Noé).  
In some cases these thrid-party licenses are **non-commerical**.
See [NOTICE](./NOTICE) for details.
