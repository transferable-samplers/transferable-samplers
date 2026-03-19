# Transferable Samplers

A research codebase for **sampling the Boltzmann density of molecular systems**, with a focus on transferable methods that generalise to unseen systems at inference time.
!> This is a research codebase, not a library. It is designed to be edited and hacked for rapid prototyping of novel approaches.
?> For an overview of how the codebase components fit together, see [Design](design.md)!

## Quickstart

```bash
# Clone repo
git clone https://github.com/transferable-samplers/transferable-samplers.git
cd transferable-samplers

# Create and activate a virtual environment
uv venv .venv --python=3.11
source .venv/bin/activate

# Install in editable mode
uv pip install -e .

# Install runtime dependencies
uv pip install -r requirements.txt
```

Then run your first experiment!
```bash
uv run python -m transferable_samplers.eval experiment=transferable/eval/prose_up_to_8aa_snis
```

> You must also populate `.env.example` and save as `.env` (sets `SCRATCH_DIR`).  
> **Optional:** Flash Attention (for TarFlow methods) must be installed separately, see the note at the bottom of `requirements.txt`.

For full usage including datasets, model weights, training, and fine-tuning, see [Usage](usage.md). For development setup, tooling, and tests, see [Contributing](contributing.md).

## Implemented Papers

This codebase is the official implementation of:

**Amortized Sampling with Transferable Normalizing Flows**  
[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-68448c)](https://neurips.cc/virtual/2025/loc/san-diego/poster/118702)
[![arXiv](https://img.shields.io/badge/arXiv-2508.18175-b31b1b)](https://arxiv.org/abs/2508.18175v4)

**Scalable Equilibrium Sampling with Sequential Boltzmann Generators**  
[![ICML 2025](https://img.shields.io/badge/ICML-2025-0077b6)](https://icml.cc/virtual/2025/poster/45137)
[![arXiv](https://img.shields.io/badge/arXiv-2502.18462-b31b1b)](https://arxiv.org/abs/2502.18462)

For details on reproducing paper results see [Paper Reproduction](paper-reproduction.md).

---

We additionally provide baseline implementations of:

**Transferable Boltzmann Generators**  
[![NeurIPS 2024](https://img.shields.io/badge/NeurIPS-2024-68448c)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/5035a409f5798e188079e236f437e522-Abstract-Conference.html)
[![arXiv](https://img.shields.io/badge/arXiv-2406.14426-b31b1b)](https://arxiv.org/abs/2406.14426)

More to come soon! 🚀

## Citation

If you use this codebase, please cite:
```bibtex
@inproceedings{
tan2025amortized,
title={Amortized Sampling with Transferable Normalizing Flows},
author={Charlie B. Tan and Majdi Hassan and Leon Klein and Saifuddin Syed and Dominique Beaini and Michael M. Bronstein and Alexander Tong and Kirill Neklyudov},
booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
year={2025},
url={https://openreview.net/forum?id=JenfC3ovzU}
}
```

## Acknowledgements

We thank HuggingFace for hosting the [ManyPeptidesMD](https://huggingface.co/datasets/transferable-samplers/many-peptides-md) dataset!

## License

The core of this repository is licensed under the MIT License (see [LICENSE](https://github.com/transferable-samplers/transferable-samplers/blob/main/LICENSE)).
Some files include adaptations of third-party code under other licenses (Apple, NVIDIA, Klein & Noé).
In some cases, these third-party licenses are **non-commercial**.
See [NOTICE](https://github.com/transferable-samplers/transferable-samplers/blob/main/NOTICE) for details.
