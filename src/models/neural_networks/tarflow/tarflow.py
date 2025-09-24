# -------------------------------------------------------------------------
# Adapted from
# https://github.com/apple/ml-tarflow/blob/main/transformer_flow.py
# Licensed under Apple License
# -------------------------------------------------------------------------
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
# 
# IMPORTANT:  This Apple software is supplied to you by Apple
# Inc. ("Apple") in consideration of your agreement to the following
# terms, and your use, installation, modification or redistribution of
# this Apple software constitutes acceptance of these terms.  If you do
# not agree with these terms, please do not use, install, modify or
# redistribute this Apple software.
# 
# In consideration of your agreement to abide by the following terms, and
# subject to these terms, Apple grants you a personal, non-exclusive
# license, under Apple's copyrights in this original Apple software (the
# "Apple Software"), to use, reproduce, modify and redistribute the Apple
# Software, with or without modifications, in source and/or binary forms;
# provided that if you redistribute the Apple Software in its entirety and
# without modifications, you must retain this notice and the following
# text and disclaimers in all such redistributions of the Apple Software.
# Neither the name, trademarks, service marks or logos of Apple Inc. may
# be used to endorse or promote products derived from the Apple Software
# without specific prior written permission from Apple.  Except as
# expressly stated in this notice, no other rights or licenses, express or
# implied, are granted by Apple herein, including but not limited to any
# patent rights that may be infringed by your derivative works or by other
# works in which the Apple Software may be incorporated.
# 
# The Apple Software is provided by Apple on an "AS IS" basis.  APPLE
# MAKES NO WARRANTIES, EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION
# THE IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE, REGARDING THE APPLE SOFTWARE OR ITS USE AND
# OPERATION ALONE OR IN COMBINATION WITH YOUR PRODUCTS.
# 
# IN NO EVENT SHALL APPLE BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL
# OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) ARISING IN ANY WAY OUT OF THE USE, REPRODUCTION,
# MODIFICATION AND/OR DISTRIBUTION OF THE APPLE SOFTWARE, HOWEVER CAUSED
# AND WHETHER UNDER THEORY OF CONTRACT, TORT (INCLUDING NEGLIGENCE),
# STRICT LIABILITY OR OTHERWISE, EVEN IF APPLE HAS BEEN ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
# -------------------------------------------------------------------------
# Modifications Copyright (c) 2025 transferable-samplers contributors
# Licensed under the MIT License (see LICENSE in the repository root).
# -------------------------------------------------------------------------

import torch
import torch.nn as nn

from src.models.neural_networks.embedder import SinusoidalEmbedding
from src.models.neural_networks.tarflow.adaptive_blocks import AdaptiveAttnAndTransition
from src.models.neural_networks.tarflow.attention import Attention, AttentionBlock

MAX_SEQ_LEN = 512


def write_tensor_to_txt(tensor: torch.Tensor, filename: str, linesize: int = 3):
    """HELPFUL FOR DEBUGGING: write a tensor to a text file for easy diffs"""
    tensor = tensor.flatten()
    with open("0DEBUG_" + filename, "w") as f:
        for j in range(0, tensor.size(0), linesize):
            triplet = tensor[j : j + linesize].tolist()
            line = " ".join(f"{x:.6f}" for x in triplet)
            f.write(line + "\n")


class PermutationFromDict(torch.nn.Module):
    def __init__(self, permutation_key: str):
        super().__init__()
        self.permutation_key = permutation_key

    def forward(self, data: torch.Tensor, data_permutations_dict: dict[str, torch.Tensor], inverse: bool = False):
        assert self.permutation_key in data_permutations_dict, (
            f"Permutation key {self.permutation_key} not found in data_permutations"
        )
        permutation = data_permutations_dict[self.permutation_key]
        if inverse:
            permutation = torch.argsort(permutation)  # get inverse permutation
        permutation = permutation.unsqueeze(-1).expand(-1, -1, data.shape[-1])
        permuted_data = torch.gather(data, dim=1, index=permutation)
        return permuted_data


class MetaBlock(torch.nn.Module):
    attn_mask: torch.Tensor

    def __init__(
        self,
        in_channels: int,
        channels: int,
        num_patches: int,
        permutation: PermutationFromDict,
        num_layers: int = 1,
        head_dim: int = 64,
        expansion: int = 4,
        nvp: bool = True,
        conditional: bool = False,
        use_adapt_ln: bool = False,
        use_attn_pair_bias: bool = False,
        pair_bias_hidden_dim: int = 16,
        use_transition: bool = False,
        use_qkln: bool = False,
        dropout: float = 0.0,
        pos_embed_type: str = "learned",  # learned, sinusoidal
        debug: bool = False,
        lookahead_conditioning: bool = False,
    ):
        super().__init__()
        self.proj_in = torch.nn.Linear(in_channels, channels)
        self.lookahead_conditioning = lookahead_conditioning

        if conditional:
            if not lookahead_conditioning:
                self.proj_cond = torch.nn.Linear(channels, channels)
            else:
                self.proj_cond = torch.nn.Sequential(
                    torch.nn.Linear(channels * 2, channels),
                    torch.nn.GELU(),
                    torch.nn.Linear(channels, channels),
                )

        if use_attn_pair_bias:
            num_heads = channels // head_dim
            # only a single projection for each block - we don't update cdists within the block
            # so i think this makes sense to just project once - you could learn a different projection for each layer
            # but i don't think this will be worthwhile
            self.pair_proj = torch.nn.Sequential(
                torch.nn.Linear(1, pair_bias_hidden_dim),
                torch.nn.GELU(),
                # needs projecting to num_heads as each head has its own attn_mask
                torch.nn.Linear(pair_bias_hidden_dim, num_heads, bias=False),  # softmax is invariant to bias
            )

            # Scale the weights of the MLP layers - to slow down "switching on of learned mask"
            with torch.no_grad():
                self.pair_proj[0].weight.mul_(1e-3)
                self.pair_proj[-1].weight.mul_(1e-9)

        self.use_attn_pair_bias = use_attn_pair_bias

        if pos_embed_type == "learned":
            if debug:
                # if debug use a larger value for the position embedding to make it easier to see borkage
                self.pos_embed = torch.nn.Parameter(torch.randn(num_patches, channels) * 1e-1)
            else:
                self.pos_embed = torch.nn.Parameter(torch.randn(num_patches, channels) * 1e-2)
        elif pos_embed_type == "sinusoidal":
            self.pos_embed = SinusoidalEmbedding(embed_size=channels, div_value=10000.0)(torch.arange(MAX_SEQ_LEN))
            self.pos_embed_scale = torch.nn.Parameter(torch.ones(1) * 1e-2)
        else:
            raise ValueError(f"Unknown pos_embed_type: {pos_embed_type}. Use 'learned' or 'sinusoidal'.")
        self.pos_embed_type = pos_embed_type

        attn_block = AdaptiveAttnAndTransition if use_adapt_ln else AttentionBlock
        self.attn_blocks = torch.nn.ModuleList(
            [
                attn_block(
                    channels=channels,
                    head_channels=head_dim,
                    expansion=expansion,
                    use_qkln=use_qkln,
                    use_attn_pair_bias=use_attn_pair_bias,
                    use_transition=use_transition,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.nvp = nvp
        output_dim = in_channels * 2 if nvp else in_channels
        self.proj_out = torch.nn.Linear(channels, output_dim)
        if debug:
            self.proj_out.weight.data = self.proj_out.weight.data * 1e-1
        else:
            self.proj_out.weight.data.fill_(0.0)
        self.permutation = permutation
        self.register_buffer("attn_mask", torch.tril(torch.ones(num_patches, num_patches)))
        self.in_channels = in_channels

    def forward(
        self,
        x: torch.Tensor,
        permutations: dict[str, torch.Tensor],
        cond: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_in = self.permutation(x, permutations)  # store permuted input for later

        x = self.proj_in(x)
        x = self.permutation(x, permutations)

        # no permutation on pos_embed - it encodes sequence position AFTER permutation
        pos_embed = self.pos_embed[: x.shape[1]]
        if self.pos_embed_type == "sinusoidal":
            pos_embed = pos_embed.to(x.device) * self.pos_embed_scale.to(x.device)  # learnable scale for sinusoid
        x = x + pos_embed

        if cond is not None:
            cond = self.permutation(cond, permutations)
            if self.lookahead_conditioning:
                lookahead_cond = torch.cat(
                    [cond[:, 1:], torch.zeros_like(cond[:, :1])], dim=1
                )  # shift back one token w/ zero pad
                cond = torch.cat([cond, lookahead_cond], dim=-1)  # concatenate the two
            cond_emb = self.proj_cond(cond)
        else:
            cond_emb = None

        pair_emb = None
        if self.use_attn_pair_bias:
            with torch.no_grad():  # don't want to backprop through this
                # pairwise distance matrix
                dist_matrix = torch.cdist(x_in, x_in)[..., None]
            pair_emb = self.pair_proj(dist_matrix)

        attn_mask = self.attn_mask
        if mask is not None:
            assert mask.shape[:1] == x.shape[:1], (
                f"First two dimensions of mask {mask.shape[:1]} and x {x.shape[:1]} do not match"
            )

            # WARNING there was a permutation of mask here but i can't see what it would do TODO
            attn_mask = attn_mask.unsqueeze(0)
            attn_mask = attn_mask * mask[..., None]
            attn_mask = attn_mask.unsqueeze(1)

        attn_mask = attn_mask[..., : x.shape[1], : x.shape[1]]
        for block in self.attn_blocks:
            x = block(x, cond=cond_emb, pair=pair_emb, mask=mask, attn_mask=attn_mask)
            if mask is not None:
                assert x[torch.where(mask == 0)].sum() == 0, "Masked positions are nonzero"

        x = self.proj_out(x)
        x = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)  # shift one token w/ zero pad
        x = x * mask[..., None] if mask is not None else x  # apply mask if provided

        if self.nvp:
            xa, xb = x.chunk(2, dim=-1)
        else:
            xb = x
            xa = torch.zeros_like(x)

        tokenization_map = permutations.get("tokenization_map", None)
        if tokenization_map is not None:
            tokenization_mask = (tokenization_map != -1).float().repeat_interleave(3, dim=-1)  # TODO hardcode
            tokenization_mask = self.permutation(tokenization_mask, permutations)
            xb = xb * tokenization_mask
            xa = xa * tokenization_mask
        scale = (-xa.float()).exp().type(xa.dtype)
        x_out = self.permutation((x_in - xb) * scale, permutations, inverse=True)

        if tokenization_map is not None:
            data_dim = (tokenization_map != -1).int().sum(
                dim=[1, 2]
            ) * 3  # this will inherently account for full padded residue tokens # TODO could be better
        elif mask is not None:
            data_dim = mask.sum(dim=-1) * 3  # TODO ugly and makes assumptions
        else:
            data_dim = x.shape[1] * 3  # assume all tokens are valid

        logdet = -xa.sum(dim=[1, 2]) / data_dim

        return x_out, logdet

    def reverse_step(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        pos_embed: torch.Tensor,
        i: int,
        attn_temp: float = 1.0,  # TODO remove?
        which_cache: str = "cond",  # TODO remove?
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_in = x.clone()
        x = self.proj_in(x_in[:, i : i + 1]) + pos_embed[:, i : i + 1]

        if cond is not None:
            if self.lookahead_conditioning:
                lookahead_cond = torch.cat(
                    [cond[:, 1:], torch.zeros_like(cond[:, :1])], dim=1
                )  # shift back one token w/ zero pad
                cond = torch.cat([cond, lookahead_cond], dim=-1)  # concatenate the two
            cond_in = cond[:, i : i + 1]
            cond_emb = self.proj_cond(cond_in)
        else:
            cond_emb = None

        pair_emb = None
        if self.use_attn_pair_bias:
            # pairwise distance row
            with torch.no_grad():  # don't want to backprop through this
                dist_matrix = torch.cdist(x_in[:, : i + 1], x_in[:, : i + 1])[..., None]
                dist_row = dist_matrix[:, i : i + 1]
            pair_emb = self.pair_proj(dist_row)

        for block in self.attn_blocks:
            x = block(
                x, cond=cond_emb, pair=pair_emb, mask=None, attn_mask=None, attn_temp=attn_temp, which_cache=which_cache
            )  # here we use kv caching, so no attn_mask

        x = self.proj_out(x)

        if self.nvp:
            xa, xb = x.chunk(2, dim=-1)
        else:
            xb = x
            xa = torch.zeros_like(x)

        return xa, xb

    def set_sample_mode(self, flag: bool = True):
        for m in self.modules():
            if isinstance(m, Attention):
                m.sample = flag
                m.k_cache = {"cond": [], "uncond": []}
                m.v_cache = {"cond": [], "uncond": []}

    def reverse(
        self,
        x: torch.Tensor,
        permutations: dict[str, torch.Tensor],
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.permutation(x, permutations)

        # no permutation on pos_embed - it encodes sequence position AFTER permutation
        pos_embed = self.pos_embed[: x.shape[1]][None, ...]
        if self.pos_embed_type == "sinusoidal":
            pos_embed = pos_embed.to(x.device) * self.pos_embed_scale.to(x.device)  # learnable scale for sinusoid

        if cond is not None:
            cond = self.permutation(cond, permutations)

        self.set_sample_mode(True)
        xs = [x[:, i] for i in range(x.size(1))]
        tokenization_map = permutations.get("tokenization_map", None)
        if tokenization_map is not None:
            tokenization_mask = (tokenization_map != -1).float().repeat_interleave(3, dim=-1)  # TODO hardcode
            tokenization_mask = self.permutation(tokenization_mask, permutations)
            tokenization_masks = [tokenization_mask[:, i] for i in range(tokenization_mask.size(1))]
        for i in range(x.size(1) - 1):
            za, zb = self.reverse_step(x, cond, pos_embed, i, which_cache="cond")
            if tokenization_map is not None:
                zb = zb * tokenization_masks[i + 1]
                za = za * tokenization_masks[i + 1]
            scale = za[:, 0].float().exp().type(za.dtype)  # get rid of the sequence dimension
            xs[i + 1] = xs[i + 1] * scale + zb[:, 0]
            x = torch.stack(xs, dim=1)

        self.set_sample_mode(False)
        x = self.permutation(x, permutations, inverse=True)

        return x


class TarFlow(torch.nn.Module):
    def __init__(
        self,
        input_dimension: int,
        max_num_tokens: int,
        channels: int,
        num_blocks: int,
        layers_per_block: int,
        head_dim: int = 64,
        use_adapt_ln: bool = False,
        use_attn_pair_bias: bool = False,
        pair_bias_hidden_dim: int = 16,
        use_transition: bool = False,
        use_qkln: bool = False,
        dropout: float = 0.0,
        permutation_keys: list[str] = [
            "n2c_residue-by-residue_standard_group-by-group",
            "n2c_residue-by-residue_standard_group-by-group_flip",
        ],  # defaults to SBG
        cond_embed: nn.Module | None = None,  # TODO don't like name, could make a proper subclass
        pos_embed_type: str = "learned",  # learned, sinusoidal
        nvp: bool = True,
        debug: bool = False,  # stops the weight initialization from being zero so tokens are not all the same
        lookahead_conditioning: bool = False,
    ):
        super().__init__()
        self.input_dimension = input_dimension
        permutation_keys = list(permutation_keys) * (
            num_blocks // len(permutation_keys) + 1
        )  # repeat to match num_blocks
        self.conditional = False if cond_embed is None else True
        self.cond_embed = cond_embed
        self.debug = debug

        blocks = []
        for i in range(num_blocks):
            blocks.append(
                MetaBlock(
                    input_dimension,
                    channels,
                    max_num_tokens,
                    PermutationFromDict(permutation_keys[i]),
                    layers_per_block,
                    head_dim=head_dim,
                    nvp=nvp,
                    use_adapt_ln=use_adapt_ln,
                    use_attn_pair_bias=use_attn_pair_bias,
                    pair_bias_hidden_dim=pair_bias_hidden_dim,
                    use_transition=use_transition,
                    use_qkln=use_qkln,
                    dropout=dropout,
                    conditional=self.conditional,
                    pos_embed_type=pos_embed_type,
                    debug=debug,
                    lookahead_conditioning=lookahead_conditioning,
                )
            )
        self.blocks = torch.nn.ModuleList(blocks)

    def forward(
        self,
        x: torch.Tensor,
        permutations: dict[str, torch.Tensor],
        encodings: dict[str, torch.Tensor] | None = None,
        mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mask is not None:
            assert mask.ndim == 2, "Mask should be 2D"
            assert torch.all(x.sum(dim=-1)[mask == 0] == 0), "x is not zero where mask is zero"
            mask = mask.view(x.shape[0], -1)  # needs to be this shape for embedder

        if self.conditional:
            assert encodings is not None, "encodings must be provided for conditional model."
            if mask is not None:
                for key in encodings.keys():
                    if not key == "seq_len":  # seq_len is not a tensor, so we don't check it
                        assert torch.all(encodings[key][mask == 0] == 0), f"{key} is not zero where mask is zero"
            # (batch_size, seq_len, channels)
            cond = self.cond_embed(**encodings, mask=mask)
        else:
            cond = None

        logdets = torch.zeros((), device=x.device)

        for block in self.blocks:
            x, logdet = block(x, permutations, cond=cond, mask=mask)
            logdets = logdets + logdet

        return x, logdets

    def reverse(
        self,
        x: torch.Tensor,
        permutations: dict[str, torch.Tensor],
        encodings: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """No masking in reverse since we assume the model generates a single peptide system as a time."""

        if self.conditional:
            assert encodings is not None, "encodings must be provided for conditional model."
            assert x.shape[1] == encodings["aa_type"].shape[1], "x and encodings do not match"

            for key in ["atom_type", "aa_type", "aa_pos"]:
                if key in encodings:
                    if not key == "seq_len":  # seq_len is single value for each batch item, so we don't check it
                        assert not torch.any(encodings[key] == 0), (
                            f"{key} has padding zeros, padding not supported in reverse"
                        )
            cond = self.cond_embed(**encodings)
        else:
            cond = None

        for block in reversed(self.blocks):
            x = block.reverse(x, permutations, cond=cond)

        return x
