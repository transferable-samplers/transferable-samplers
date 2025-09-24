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

import numpy as np
import torch
import torch.nn as nn

from src.models.neural_networks.egnn.egnn import EGNN
from src.models.neural_networks.egnn.utils import remove_mean


class EGNN_dynamics_AD2_cat(nn.Module):
    def __init__(
        self,
        num_atoms,
        num_dimensions,
        channels=64,
        act_fn=torch.nn.SiLU(),
        num_layers=5,  # changed to match AD2_classical_train_tgb_full.py
        recurrent=True,
        attention=True,  # changed to match AD2_classical_train_tgb_full.py
        tanh=True,  # changed to match AD2_classical_train_tgb_full.py
        atom_encodings_filename: str = "atom_types_ecoding.npy",
        data_dir="data/alanine",
        pdb_filename="AAAAAA_310K.pdb",
        agg="sum",
        M=128,
    ):
        super().__init__()
        self._n_particles = num_atoms
        self._n_dimensions = num_dimensions
        # Initial one hot encodings of the different element types
        self.h_initial = self.get_h_initial()

        h_size = self.h_initial.size(1)
        h_size += 1  # Add time

        self.egnn = EGNN(
            in_node_nf=h_size,
            in_edge_nf=1,
            hidden_nf=channels,
            act_fn=act_fn,
            n_layers=num_layers,
            recurrent=recurrent,
            attention=attention,
            tanh=tanh,
            agg=agg,
        )

        self.edges = self._create_edges()
        self._edges_dict = {}
        # Count function calls
        self.counter = 0
        self.M = M

    def get_h_initial(self):
        if self._n_particles == 22:
            atom_types = np.arange(22)
            atom_types[[1, 2, 3]] = 2
            atom_types[[11, 12, 13]] = 12
            atom_types[[19, 20, 21]] = 20
        if self._n_particles == 33:
            atom_types = np.arange(33)
            atom_types[[1, 2, 3]] = 2
            atom_types[[9, 10, 11]] = 10
            atom_types[[19, 20, 21]] = 20
            atom_types[[29, 30, 31]] = 30
        if self._n_particles == 42:
            atom_types = np.arange(42)
            atom_types[[1, 2, 3]] = 2
            atom_types[[11, 12, 13]] = 12
            atom_types[[21, 22, 23]] = 22
            atom_types[[31, 32, 33]] = 32
            atom_types[[39, 40, 41]] = 40
        if self._n_particles == 63:
            atom_types = np.arange(63)
            atom_types[[1, 2, 3]] = 2
            atom_types[[7, 8, 9]] = 8
            atom_types[[17, 18, 19]] = 18
            atom_types[[27, 28, 29]] = 28
            atom_types[[37, 38, 39]] = 38
            atom_types[[47, 48, 49]] = 48
            atom_types[[57, 58, 59]] = 58
        h_initial = torch.nn.functional.one_hot(torch.tensor(atom_types))
        return h_initial

    def get_hidden(self):
        amino_dict = {
            "ALA": 0,
            "ARG": 1,
            "ASN": 2,
            "ASP": 3,
            "CYS": 4,
            "GLN": 5,
            "GLU": 6,
            "GLY": 7,
            "HIS": 8,
            "ILE": 9,
            "LEU": 10,
            "LYS": 11,
            "MET": 12,
            "PHE": 13,
            "PRO": 14,
            "SER": 15,
            "THR": 16,
            "TRP": 17,
            "TYR": 18,
            "VAL": 19,
        }
        atom_types = []
        amino_idx = []
        amino_types = []
        for i, amino in enumerate(self.topology.residues):
            for atom_name in amino.atoms:
                amino_idx.append(i)
                amino_types.append(amino_dict[amino.name])
                if atom_name.name[0] == "H" and atom_name.name[-1] in ("1", "2", "3"):
                    if amino_dict[amino.name] in (8, 13, 17, 18) and atom_name.name[:2] in (
                        "HE",
                        "HD",
                        "HZ",
                        "HH",
                    ):
                        pass
                    else:
                        atom_name.name = atom_name.name[:-1]
                if atom_name.name[:2] == "OE" or atom_name.name[:2] == "OD":
                    atom_name.name = atom_name.name[:-1]
                atom_types.append(atom_name.name)
        atom_types_dict = np.array([self.atom_types_encodings[atom_type] for atom_type in atom_types])
        atom_onehot = torch.nn.functional.one_hot(
            torch.tensor(atom_types_dict), num_classes=len(self.atom_types_encodings)
        )
        if self._n_particles == 53:
            num_classes = 5
        elif self._n_particles == 63:
            num_classes = 6
        amino_idx_onehot = torch.nn.functional.one_hot(torch.tensor(amino_idx), num_classes=num_classes)
        amino_types_onehot = torch.nn.functional.one_hot(torch.tensor(amino_types), num_classes=20)

        h_initial = torch.cat([amino_idx_onehot, amino_types_onehot, atom_onehot], dim=1)
        return h_initial

    def forward(self, t, x, *args, **kwargs):
        t = t.view(-1, 1)

        if t.numel() == 1:
            t = t.repeat(x.shape[0], 1)

        n_batch = x.shape[0]
        edges = self._cast_edges2batch(self.edges, n_batch, self._n_particles, device=x.device)
        edges = [edges[0], edges[1]]

        # Changed by Leon
        x = x.reshape(n_batch * self._n_particles, self._n_dimensions).clone()
        h = self.h_initial.to(x.device).reshape(1, -1)
        h = h.repeat(n_batch, 1)
        h = h.reshape(n_batch * self._n_particles, -1)

        if t.shape != (n_batch, 1):
            t = t.repeat(n_batch)
        t = t.repeat(1, self._n_particles)
        t = t.reshape(n_batch * self._n_particles, 1)

        h = torch.cat([h, t], dim=-1)
        edge_attr = torch.sum((x[edges[0]] - x[edges[1]]) ** 2, dim=1, keepdim=True)
        _, x_final = self.egnn(h, x, edges, edge_attr=edge_attr)
        vel = x_final - x

        vel = vel.view(n_batch, self._n_particles, self._n_dimensions)
        vel = remove_mean(vel)
        self.counter += 1
        return vel.view(n_batch, self._n_particles * self._n_dimensions)

    def _create_edges(self):
        rows, cols = [], []
        for i in range(self._n_particles):
            for j in range(i + 1, self._n_particles):
                rows.append(i)
                cols.append(j)
                rows.append(j)
                cols.append(i)
        return [torch.LongTensor(rows), torch.LongTensor(cols)]

    def _cast_edges2batch(self, edges, n_batch, n_nodes, device):
        if n_batch not in self._edges_dict:
            self._edges_dict = {}
            rows, cols = edges
            rows_total, cols_total = [], []
            for i in range(n_batch):
                rows_total.append(rows + i * n_nodes)
                cols_total.append(cols + i * n_nodes)
            rows_total = torch.cat(rows_total).to(device)
            cols_total = torch.cat(cols_total).to(device)

            self._edges_dict[n_batch] = [rows_total, cols_total]
        return self._edges_dict[n_batch]
