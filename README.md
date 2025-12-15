> [!IMPORTANT]
> **Critical Dataset Update**
> 
> The original 8AA TICA models within `subsampled_trajectories/*/8AA/*.npz` employed a CA-only atom selection. **These models are not valid for comparison to results in our paper.**
> 
> **Updated files (uploaded 15/12/2025)** now contain corrected models. If you previously downloaded this dataset, please re-download to ensure accurate results.
>
> Note: Codebase references to `tica_features_ca` must now be replaced with `tica_features`. **This was resolved in our codebase by [PR #26](https://github.com/transferable-samplers/transferable-samplers/pull/26).**
>
> Note: Unguarded `snapshot_download` calls will automatically redownload the relevant files when it detects a change in the repo.
> 
> We sincerely apologize for any inconvenience this may have caused.

# Transferable Samplers

Welcome to the official codebase for the following works:

**Amortized Sampling with Transferable Normalizing Flows** [*NeurIPS 2025*](https://arxiv.org/abs/2508.18175)

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

In both cases, the codebase is set up to automatically download the necessary data for training and evaluation. In the case of ManyPeptidesMD the training webdataset will by default be streamed and cached.

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

To use a locally generated checkpoint, you may pass in the argument `ckpt_path` to override the remote weights usage.

## Acknowledgements

We would like to thank Hugging Face for hosting our large ManyPeptidesMD dataset!

## License

The core of this repository is licensed under the MIT License (see [LICENSE](./LICENSE)).  
Some files include adaptation of third-party code under other licenses (Apple, Meta, NVIDIA, Klein & Noé).  
In some cases, these thrid-party licenses are **non-commerical**.
See [NOTICE](./NOTICE) for details.
