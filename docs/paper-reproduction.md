# Paper Reproduction

All experiment configs are under `configs/experiment/`. Evaluation configs default to downloading pretrained weights from Hugging Face — see [Usage](usage.md) for how to override with local checkpoints.

By default each evaluation runs over all systems defined in the data config. See [Evaluation](usage.md#sampling) in Usage for details on running only a single system per process - in practice it is often more convenient to run each evaluation separately and aggregate across systems afterwards.

---

## Amortized Sampling with Transferable Normalizing Flows
[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-68448c)](https://neurips.cc/virtual/2025/loc/san-diego/poster/118702)
[![arXiv](https://img.shields.io/badge/arXiv-2508.18175-b31b1b)](https://arxiv.org/abs/2508.18175v4)

### Training

```bash
# Train Prose on ManyPeptidesMD
uv run python -m transferable_samplers.train experiment=transferable/train/prose_up_to_8aa

# Train TarFlow on ManyPeptidesMD
uv run python -m transferable_samplers.train experiment=transferable/train/tarflow_up_to_8aa

# Train ECNF++ on ManyPeptidesMD (up to 4aa only)
uv run python -m transferable_samplers.train experiment=transferable/train/ecnf++_up_to_4aa
```

### Table 2 — SNIS (10k energy evaluations)


```bash
# Prose SNIS
uv run python -m transferable_samplers.eval experiment=transferable/eval/prose_up_to_8aa_snis \
  callbacks.sampling_evaluation.sampler.num_samples=10000

# TarFlow SNIS
uv run python -m transferable_samplers.eval experiment=transferable/eval/tarflow_up_to_8aa_snis \
  callbacks.sampling_evaluation.sampler.num_samples=10000

# ECNF++ SNIS
uv run python -m transferable_samplers.eval experiment=transferable/eval/ecnf++_up_to_4aa_snis
```

### Table 4 — SNIS / SMC / Self-Improvement (1M energy evaluations)

```bash
# SNIS (1M samples)
uv run python -m transferable_samplers.eval experiment=transferable/eval/prose_up_to_8aa_snis

# SMC with ULA (continuous)
uv run python -m transferable_samplers.eval experiment=transferable/eval/prose_up_to_8aa_ula

# SMC with MALA (discrete)
uv run python -m transferable_samplers.eval experiment=transferable/eval/prose_up_to_8aa_mala

# Self-improving SNIS
uv run python -m transferable_samplers.train experiment=transferable/fine-tune/prose_up_to_8aa_self_improve
```

## Scalable Equilibrium Sampling with Sequential Boltzmann Generators
[![ICML 2025](https://img.shields.io/badge/ICML-2025-0077b6)](https://icml.cc/virtual/2025/poster/45137)
[![arXiv](https://img.shields.io/badge/arXiv-2502.18462-b31b1b)](https://arxiv.org/abs/2502.18462)


### Table 2 — (Ace-A-Nme, AAA)

```bash
# ECNF++ — Ace-A-Nme
uv run python -m transferable_samplers.train experiment=single_system/train/ecnf++_Ace-A-Nme
uv run python -m transferable_samplers.eval experiment=single_system/eval/ecnf++_Ace-A-Nme_snis

# TarFlow — Ace-A-Nme
uv run python -m transferable_samplers.train experiment=single_system/train/tarflow_Ace-A-Nme
uv run python -m transferable_samplers.eval experiment=single_system/eval/tarflow_Ace-A-Nme_ula

# ECNF++ — AAA
uv run python -m transferable_samplers.train experiment=single_system/train/ecnf++_AAA
uv run python -m transferable_samplers.eval experiment=single_system/eval/ecnf++_AAA_snis

# TarFlow — AAA
uv run python -m transferable_samplers.train experiment=single_system/train/tarflow_AAA
uv run python -m transferable_samplers.eval experiment=single_system/eval/tarflow_AAA_ula
```

> To reproduce results averaged over seeds, override `hf_state_dict_path` with `_1` or `_2` (e.g. `hf_state_dict_path=single_system/ecnf++_AAA_1.pth`).

### Table 3 — Tetrapeptides (Ace-AAA-Nme, AAAAAA)

```bash
# ECNF++ — Ace-AAA-Nme
uv run python -m transferable_samplers.train experiment=single_system/train/ecnf++_Ace-AAA-Nme
uv run python -m transferable_samplers.eval experiment=single_system/eval/ecnf++_Ace-AAA-Nme_snis

# TarFlow — Ace-AAA-Nme
uv run python -m transferable_samplers.train experiment=single_system/train/tarflow_Ace-AAA-Nme
uv run python -m transferable_samplers.eval experiment=single_system/eval/tarflow_Ace-AAA-Nme_ula

# ECNF++ — AAAAAA
uv run python -m transferable_samplers.train experiment=single_system/train/ecnf++_AAAAAA
uv run python -m transferable_samplers.eval experiment=single_system/eval/ecnf++_AAAAAA_snis

# TarFlow — AAAAAA
uv run python -m transferable_samplers.train experiment=single_system/train/tarflow_AAAAAA
uv run python -m transferable_samplers.eval experiment=single_system/eval/tarflow_AAAAAA_ula
```

> To reproduce results averaged over seeds, override `hf_state_dict_path` with `_1` or `_2` (e.g. `hf_state_dict_path=single_system/ecnf++_AAA_1.pth`).

### Figure 8 — Chignolin (GYDPETGTWG)

```bash
uv run python -m transferable_samplers.train experiment=single_system/train/tarflow_GYDPETGTWG
uv run python -m transferable_samplers.eval experiment=single_system/eval/tarflow_GYDPETGTWG_ula
```
