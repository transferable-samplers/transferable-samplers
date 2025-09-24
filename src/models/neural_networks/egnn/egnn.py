# -------------------------------------------------------------------------
# Adapted from
# https://osf.io/n8vz3/
# Licensed under Creative Commons Attribution 4.0 International
# -------------------------------------------------------------------------
# Copyright (c) 2024 Leon Klein, Frank Noé.
# https://creativecommons.org/licenses/by/4.0/
# -------------------------------------------------------------------------
# Modifications Copyright (c) 2025 transferable-samplers contributors
# Licensed under the MIT License (see LICENSE in the repository root).
# -------------------------------------------------------------------------

import torch.nn as nn

from src.models.neural_networks.egnn.gcl import E_GCL
from src.models.neural_networks.egnn.timestep_embedder import TimestepEmbedder


class EGNN(nn.Module):
    def __init__(
        self,
        in_node_nf,
        in_edge_nf,
        hidden_nf,
        act_fn=nn.SiLU(),
        n_layers=4,
        recurrent=True,
        attention=False,
        norm_diff=True,
        out_node_nf=None,
        tanh=False,
        coords_range=15,
        agg="sum",
    ):
        super().__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range) / self.n_layers
        if agg == "mean":
            self.coords_range_layer = self.coords_range_layer * 19
        # self.reg = reg
        # Encoder
        # self.add_module("gcl_0", E_GCL(in_node_nf, self.hidden_nf, self.hidden_nf,
        # edges_in_d=in_edge_nf, act_fn=act_fn, recurrent=False, coords_weight=coords_weight))
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module(
                f"gcl_{i}",
                E_GCL(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    edges_in_d=in_edge_nf,
                    act_fn=act_fn,
                    recurrent=recurrent,
                    attention=attention,
                    norm_diff=norm_diff,
                    tanh=tanh,
                    coords_range=self.coords_range_layer,
                    agg=agg,
                ),
            )

    def forward(self, h, x, edges, edge_attr=None, node_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, x, _ = self._modules[f"gcl_{i}"](
                h,
                edges,
                x,
                edge_attr=edge_attr,
                node_mask=node_mask,
                edge_mask=edge_mask,
            )
        h = self.embedding_out(h)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h, x


class TEGNN(EGNN):
    def __init__(
        self,
        in_node_nf,
        in_edge_nf,
        hidden_nf,
        act_fn=nn.SiLU(),
        n_layers=4,
        recurrent=True,
        attention=False,
        norm_diff=True,
        out_node_nf=None,
        tanh=False,
        coords_range=15,
        agg="sum",
    ):
        super().__init__(
            in_node_nf,
            in_edge_nf,
            hidden_nf,
            act_fn=act_fn,
            n_layers=n_layers,
            recurrent=recurrent,
            attention=attention,
            norm_diff=norm_diff,
            out_node_nf=out_node_nf,
            tanh=tanh,
            coords_range=coords_range,
            agg=agg,
        )
        self.t_embedder = TimestepEmbedder(hidden_size=self.hidden_nf)

    def forward(self, h, x, t, edges, edge_attr=None, node_mask=None, edge_mask=None):
        # h is shape (num_atoms, n_features)
        # t_emb is shape (n_batch * n_atoms, n_hidden)
        t_emb = self.t_embedder(t).unsqueeze(1).repeat(1, h.shape[0], 1).reshape(-1, self.hidden_nf)
        # h is shape (1, num_atoms, n_hidden)
        h = self.embedding(h).unsqueeze(0).repeat(t.shape[0], 1, 1).reshape(-1, self.hidden_nf)
        for i in range(0, self.n_layers):
            h = h + t_emb
            h, x, _ = self._modules[f"gcl_{i}"](
                h,
                edges,
                x,
                edge_attr=edge_attr,
                node_mask=node_mask,
                edge_mask=edge_mask,
            )
        h = self.embedding_out(h)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h, x
