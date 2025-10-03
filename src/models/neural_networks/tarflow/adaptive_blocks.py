# -------------------------------------------------------------------------
# Adapted from
# https://github.com/NVIDIA-Digital-Bio/proteina/tree/main/proteinfoundation/nn
# Licensed under NVIDIA License
# -------------------------------------------------------------------------
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
# https://github.com/NVIDIA-Digital-Bio/proteina/blob/main/LICENSE
# -------------------------------------------------------------------------
# Modifications Copyright (c) 2025 transferable-samplers contributors
# Licensed under the MIT License (see LICENSE in the repository root).
# -------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from src.models.neural_networks.tarflow.attention import AttentionBlock
except ImportError:
    from attention import AttentionBlock


# Code adapted from Lucidrain's implementation of AF3
# https://github.com/lucidrains/alphafold3-pytorch
class AdaptiveLayerNorm(torch.nn.Module):
    """Adaptive layer norm layer, where scales and biases are learned from some
    conditioning variables."""

    def __init__(self, *, channels: int, channels_cond: int):
        super().__init__()
        self.norm = torch.nn.LayerNorm(channels, elementwise_affine=False)
        self.norm_cond = torch.nn.LayerNorm(channels_cond)

        self.to_gamma = torch.nn.Sequential(torch.nn.Linear(channels_cond, channels), torch.nn.Sigmoid())
        self.to_beta = torch.nn.Linear(channels_cond, channels, bias=False)

    def forward(self, x, cond, mask):
        """
        Args:
            x: input representation, shape [*, dim]
            cond: conditioning variables, shape [*, dim_cond]
            mask: binary, shape [*]

        Returns:
            Representation after adaptive layer norm, shape as input representation [*, dim].
        """
        normed = self.norm(x)
        normed_cond = self.norm_cond(cond)

        gamma = self.to_gamma(normed_cond)
        beta = self.to_beta(normed_cond)
        out = normed * gamma + beta
        return out * mask[..., None]


# Code adapted from Lucidrain's implementation of AF3
# https://github.com/lucidrains/alphafold3-pytorch
class AdaptiveLayerNormOutputScale(torch.nn.Module):
    """Adaptive scaling of a representation given conditioning variables."""

    def __init__(self, *, channels, channels_cond, adaln_zero_bias_init_value=-2.0):
        super().__init__()

        adaln_zero_gamma_linear = torch.nn.Linear(channels_cond, channels)
        torch.nn.init.zeros_(adaln_zero_gamma_linear.weight)
        torch.nn.init.constant_(adaln_zero_gamma_linear.bias, adaln_zero_bias_init_value)

        self.to_adaln_zero_gamma = torch.nn.Sequential(adaln_zero_gamma_linear, torch.nn.Sigmoid())

    def forward(self, x, cond, mask):
        """
        Args:
            x: input sequence, shape [*, dim]
            cond: conditioning variables, shape [*, dim_cond]
            mask: binary, shape [*]

        Returns:
            Scaled input, shape [*, dim].
        """
        gamma = self.to_adaln_zero_gamma(cond)  # [*, dim]
        return x * gamma * mask[..., None]


# Code adapted from Lucidrain's implementation of AF3
# https://github.com/lucidrains/alphafold3-pytorch
class SwiGLU(torch.nn.Module):
    """SwiGLU layer."""

    def forward(self, x):
        """
        Args:
            x: input tensor, shape [..., d]

        Returns:
            Tensor of shape [..., d//2].
        """
        x, gates = x.chunk(2, dim=-1)
        return F.silu(gates) * x


# Code adapted from Lucidrain's implementation of AF3
# https://github.com/lucidrains/alphafold3-pytorch
class Transition(torch.nn.Module):
    """Transition layer."""

    def __init__(self, channels, expansion_factor=4, layer_norm=False):
        super().__init__()

        channels_inner = int(channels * expansion_factor)

        self.use_layer_norm = layer_norm
        if self.use_layer_norm:
            self.ln = torch.nn.LayerNorm(channels)

        self.swish_linear = torch.nn.Sequential(
            torch.nn.Linear(channels, channels_inner * 2, bias=False),
            SwiGLU(),
        )
        self.linear_out = torch.nn.Linear(channels_inner, channels, bias=False)

    def forward(self, x, mask):
        """
        Args:
            x: Input sequence representation, shape [b, n, dim]
            mask: binary, shape [b, n]

        Returns:
            Updated sequence representation, shape [b, n, dim]
        """
        if self.use_layer_norm:
            x = self.ln(x)
        x = self.linear_out(self.swish_linear(x))
        return x * mask[..., None]


class MultiHeadAttentionADALN(nn.Module):
    """
    Adapted from Proteina:
        https://github.com/NVIDIA-Digital-Bio/proteina/blob/main/proteinfoundation/nn/protein_transformer.py

    Pair biased multi-head self-attention with adaptive layer norm applied to input
    and adaptive scaling applied to output.
    """

    def __init__(
        self,
        channels: int = 128,
        head_channels: int = 64,
        use_qkln: bool = False,
        use_attn_pair_bias: bool = False,
        dropout: float = 0.0,
        expansion: int = 4,
    ):
        super().__init__()

        assert channels % head_channels == 0, "in_channels must be divisible by head_channels"
        self.adaln = AdaptiveLayerNorm(channels=channels, channels_cond=channels)
        self.mha = AttentionBlock(
            channels=channels,
            head_channels=head_channels,
            expansion=expansion,
            use_qkln=use_qkln,
            use_attn_pair_bias=use_attn_pair_bias,
            dropout=dropout,
        )
        self.scale_output = AdaptiveLayerNormOutputScale(channels=channels, channels_cond=channels)

    def forward(self, x, cond, mask, attn_mask, pair=None, attn_temp: float = 1.0, which_cache: str = "cond"):
        """
        Args:
            x: Input sequence representation, shape [b, n, channels]
            cond: Conditioning variables, shape [b, n, channels]
            mask: Binary mask, shape [b, n]

        Returns:
            Updated sequence representation, shape [b, n, channels].
        """
        x = self.adaln(x, cond, mask)
        # Following exact scheme of proteina. Not adding conditioning to x
        # If cond were not None, then x = x + cond inside mha
        # Which we don't want as conditoning is applied through adaln and scale_output
        # TODO I think in long run we should factor out the condiioning from the mhd
        # could just have an "AdditiveConditioningAttentionBlock" module that adds the conditioning
        x = self.mha(
            x, cond=None, pair=pair, mask=mask, attn_mask=attn_mask, attn_temp=attn_temp, which_cache=which_cache
        )
        x = self.scale_output(x, cond, mask)
        return x * mask[..., None]  # [b, n, channels]


class TransitionADALN(torch.nn.Module):
    """Transition layer with adaptive layer norm applied to input and adaptive
    scaling applied to output."""

    def __init__(self, channels, channels_cond, expansion_factor=4):
        super().__init__()
        self.adaln = AdaptiveLayerNorm(channels=channels, channels_cond=channels_cond)
        self.transition = Transition(channels=channels, expansion_factor=expansion_factor, layer_norm=False)
        self.scale_output = AdaptiveLayerNormOutputScale(channels=channels, channels_cond=channels_cond)

    def forward(self, x, cond, mask):
        """
        Args:
            x: Input sequence representation, shape [b, n, dim]
            cond: conditioning variables, shape [b, n, dim_cond]
            mask: binary mask, shape [b, n]

        Returns:
            Updated sequence representation, shape [b, n, dim]
        """
        x = self.adaln(x, cond, mask)  # [b, n, dim]
        x = self.transition(x, mask)  # [b, n, dim]
        x = self.scale_output(x, cond, mask)  # [b, n, dim]
        return x * mask[..., None]  # [b, n, dim]


class AdaptiveAttnAndTransition(torch.nn.Module):
    """Layer that applies mha and transition to a sequence representation. Both layers are their adaptive versions
    which rely on conditining variables (see above).

    Args:
        dim_token: Token dimension in sequence representation.
        dim_pair: Dimension of pair representation.
        nheads: Number of attention heads.
        dim_cond: Dimension of conditioning variables.
        residual_mha: Whether to use a residual connection in the mha layer.
        residual_transition: Whether to use a residual connection in the transition layer.
        parallel_mha_transition: Whether to run mha and transition in parallel or sequentially.
        use_attn_pair_bias: Whether to use a pair represnetation to bias attention.
        use_qkln: Whether to use layer norm on keyus and queries for attention.
        dropout: droput use in the self-attention layer.
    """

    def __init__(
        self,
        channels: int = 128,
        head_channels: int = 64,
        residual_mha: bool = True,
        residual_transition: bool = True,
        use_qkln: bool = False,  # false for consistency with tarflow.py
        use_attn_pair_bias: bool = False,  # false for consistency with tarflow.py
        use_transition: bool = True,
        dropout=0.0,
        expansion=4,
    ):
        super().__init__()

        assert channels % head_channels == 0, "in_channels must be divisible by head_dim"
        self.residual_mha = residual_mha
        self.residual_transition = residual_transition
        self.use_transition = use_transition

        self.mha = MultiHeadAttentionADALN(
            channels=channels,
            head_channels=head_channels,
            use_qkln=use_qkln,
            use_attn_pair_bias=use_attn_pair_bias,
            dropout=dropout,
            expansion=expansion,
        )

        if use_transition:
            self.transition = TransitionADALN(channels=channels, channels_cond=channels, expansion_factor=expansion)

    def _apply_mha(self, x, cond, mask, pair=None, attn_mask=None, attn_temp: float = 1.0, which_cache: str = "cond"):
        x_attn = self.mha(
            x, cond=cond, pair=pair, mask=mask, attn_mask=attn_mask, attn_temp=attn_temp, which_cache=which_cache
        )
        if self.residual_mha:
            x_attn = x_attn + x
        return x_attn * mask[..., None]

    def _apply_transition(self, x, cond, mask):
        x_tr = self.transition(x, cond, mask)
        if self.residual_transition:
            x_tr = x_tr + x
        return x_tr * mask[..., None]

    def forward(self, x, cond, pair=None, mask=None, attn_mask=None, attn_temp: float = 1.0, which_cache: str = "cond"):
        """
        Args:
            x: Input sequence representation, shape [b, n, dim_token]
            cond: conditioning variables, shape [b, n, dim_cond]
            mask: binary mask, shape [b, n]

        Returns:
            Updated sequence representation, shape [b, n, dim].
        """
        if mask is None:
            mask = torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)

        x = x * mask[..., None]
        x = self._apply_mha(
            x, cond=cond, pair=pair, mask=mask, attn_mask=attn_mask, attn_temp=attn_temp, which_cache=which_cache
        )
        if self.use_transition:
            x = self._apply_transition(x, cond, mask)

        return x * mask[..., None]


if __name__ == "__main__":
    # Example usage
    batch_size = 2
    num_atoms = 10
    channels = 128
    num_heads = 8

    from src.models.neural_networks.embedder import ConditionalEmbedder

    x = torch.randn((batch_size, num_atoms, channels))
    atom_type = torch.randint(0, 54, (batch_size, num_atoms))
    aa_type = torch.randint(0, 20, (batch_size, num_atoms))
    aa_pos = torch.randint(0, 2, (batch_size, num_atoms))
    mask = torch.ones((batch_size, num_atoms), dtype=torch.bool)

    embedder = ConditionalEmbedder(hidden_dim=channels)
    cond = embedder(atom_type, aa_type, aa_pos, mask)

    adapt_layernorm = AdaptiveAttnAndTransition(in_channels=channels, head_channels=64)
    attn_mask = torch.tril(torch.ones((num_atoms, num_atoms), dtype=torch.bool))
    output = adapt_layernorm(x, cond, mask, attn_mask)
    print(output.shape)  # Should be [batch_size, num_atoms, channels]

    print(output)
    assert not torch.any(output.isnan()), "Output contains NaN values"
    assert not torch.any(output.isinf()), "Output contains inf values"

    num_atoms2 = 5
    x2 = torch.randn((batch_size, num_atoms2, channels))
    atom_type2 = torch.randint(0, 54, (batch_size, num_atoms2))
    aa_type2 = torch.randint(0, 20, (batch_size, num_atoms2))
    aa_pos2 = torch.randint(0, 2, (batch_size, num_atoms2))
    mask2 = torch.ones((batch_size, num_atoms2), dtype=torch.bool)

    # pad atom_type2, aa_type2, aa_pos2 to match num_atoms
    x2 = torch.concat([x2, torch.zeros((batch_size, num_atoms - num_atoms2, channels))], dim=1)
    atom_type2 = torch.concat([atom_type2, torch.zeros((batch_size, num_atoms - num_atoms2), dtype=torch.long)], dim=1)
    aa_type2 = torch.concat([aa_type2, torch.zeros((batch_size, num_atoms - num_atoms2), dtype=torch.long)], dim=1)
    aa_pos2 = torch.concat([aa_pos2, torch.zeros((batch_size, num_atoms - num_atoms2), dtype=torch.long)], dim=1)
    mask2 = torch.concat([mask2, torch.zeros((batch_size, num_atoms - num_atoms2), dtype=torch.bool)], dim=1)

    # concat x and x2, atom_type and atom_type2, etc
    x = torch.concat([x, x2], dim=0)
    atom_type = torch.concat([atom_type, atom_type2], dim=0)
    aa_type = torch.concat([aa_type, aa_type2], dim=0)
    aa_pos = torch.concat([aa_pos, aa_pos2], dim=0)
    mask = torch.concat([mask, mask2], dim=0)
    cond = embedder(atom_type, aa_type, aa_pos)
    attn_mask = torch.tril(torch.ones((x.shape[1], x.shape[1]), dtype=torch.bool))
    attn_mask = attn_mask[None, ...].repeat(x.shape[0], 1, 1)
    attn_mask = mask[..., None] * attn_mask
    attn_mask = attn_mask.unsqueeze(1)
    output = adapt_layernorm(x, cond, mask, attn_mask)

    print(output.shape)  # Should be [batch_size*2, num_atoms, channels]
    print(output)
    assert not torch.any(output.isnan()), "Output contains NaN values"
    assert not torch.any(output.isinf()), "Output contains inf values"
