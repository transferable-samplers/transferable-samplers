# file adapted from TBG repo

import torch
import torch.nn as nn

from src.models.neural_networks.egnn.egnn import EGNN

# TODO remove hardcode
NUM_ATOM_TYPE = 54
NUM_AA_TYPE = 20


def remove_mean_with_mask(x, node_mask):
    N = node_mask.sum(1, keepdims=True)

    mean = torch.sum(x, dim=1, keepdim=True) / N
    x = x - mean * node_mask
    return x


class EGNNDynamicsTransferableMD(nn.Module):
    def __init__(
        self,
        num_atoms,
        num_dimensions,
        num_aa_min,
        num_aa_max,
        channels,
        num_layers,
        act_fn=torch.nn.SiLU(),
        recurrent=True,
        attention=True,
        tanh=True,
        agg="sum",
        *args,
        **kwargs,
    ):
        # atom type + amino acid index + amino acid type + time
        encoding_dim = NUM_ATOM_TYPE + num_aa_max + NUM_AA_TYPE + 1
        assert num_aa_min <= num_aa_max, "num_aa_min should be less than or equal to num_aa_max"
        num_valid_seq_lens = num_aa_max - num_aa_min + 1
        if num_valid_seq_lens > 1:
            encoding_dim += num_valid_seq_lens  # add for sequence length encoding

        super().__init__()
        self.egnn = EGNN(
            in_node_nf=encoding_dim,
            in_edge_nf=1,
            hidden_nf=channels,
            act_fn=act_fn,
            n_layers=num_layers,
            recurrent=recurrent,
            attention=attention,
            tanh=tanh,
            agg=agg,
        )

        self.num_aa_max = num_aa_max
        self.num_valid_seq_lens = num_valid_seq_lens  # number of valid amino acids (excluding padding)

        # maximum number of possible particles
        self.num_atoms = num_atoms
        self.num_dimensions = num_dimensions
        self.edges_dict = {}

        # Count function calls
        self.counter = 0

    def forward(self, t, x, encodings, node_mask=None):
        # TODO resolve this another way - comes up in Jacobian where x is sliced
        # but the partial still gives full encoding
        if x.shape[0] == 1 and encodings["atom_type"].shape[0] > 1:
            assert all(torch.equal(encoding[0], encoding[1]) for encoding in encodings.values()), (
                "Encodings are not equal across batch dimension"
            )
            encodings = {key: value[0:1] for key, value in encodings.items()}

        assert not x.shape[1] % self.num_dimensions, "x should be divisible by num_atoms"
        num_atoms = x.shape[1] // self.num_dimensions

        if node_mask is not None:
            assert node_mask.ndim == 2, "Mask should be 2D"
            assert torch.all(x.view(-1, num_atoms, self.num_dimensions).sum(dim=-1)[node_mask == 0] == 0), (
                "x is not zero where mask is zero"
            )
            assert torch.all(encodings["atom_type"][node_mask == 0] == 0), "atom_type is not zero where mask is zero"
            assert torch.all(encodings["aa_type"][node_mask == 0] == 0), "aa_type is not zero where mask is zero"
            assert torch.all(encodings["aa_pos"][node_mask == 0] == 0), "aa_pos is not zero where mask is zero"

        if node_mask is None:
            node_mask = torch.ones(x.shape[0], num_atoms, device=x.device, dtype=torch.float)
        else:
            node_mask = node_mask.float()

        # edge_mask is outer product of node_mask
        edge_mask = node_mask.unsqueeze(2) * node_mask.unsqueeze(1)  # [B, N, N]

        # Remove self-edges (diagonal)
        diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool, device=edge_mask.device)
        edge_mask = edge_mask * diag_mask.unsqueeze(0)  # broadcast over batch

        batch_size = x.shape[0]

        # Prepare edges
        edges = self.get_adj_matrix(num_atoms, batch_size, device=x.device)
        edges = [edges[0], edges[1]]

        # Reshape masks
        node_mask = node_mask.view(batch_size * num_atoms, 1)
        edge_mask = edge_mask.view(batch_size * num_atoms**2, 1)

        # Reshape x - apply node_mask
        x = x.reshape(batch_size * num_atoms, self.num_dimensions).clone() * node_mask

        # Prepare time embedding
        t = t.to(x)
        if t.shape != (batch_size, 1):
            t = t.repeat(batch_size)
        t = t.repeat(1, num_atoms)
        t = t.reshape(batch_size * num_atoms, 1) * node_mask

        # Encodings have padding tokens at zero
        # We don't want this to correspond to a dimension in the onehot encoding
        # So we subtract 1 from the encodings, and then apply a max operation to ensure no negative values
        # The padding will be masked out by node_mask later

        atom_type_encoding = torch.clamp(encodings["atom_type"] - 1, min=0)
        aa_pos_encoding = torch.clamp(encodings["aa_pos"] - 1, min=0)
        aa_type_encoding = torch.clamp(encodings["aa_type"] - 1, min=0)
        seq_len_encoding = torch.clamp(encodings["seq_len"] - self.num_aa_max + self.num_valid_seq_lens - 1, min=0)

        atom_type_onehot = torch.nn.functional.one_hot(atom_type_encoding, num_classes=NUM_ATOM_TYPE)
        amino_idx_onehot = torch.nn.functional.one_hot(aa_pos_encoding, num_classes=self.num_aa_max)
        amino_types_onehot = torch.nn.functional.one_hot(aa_type_encoding, num_classes=NUM_AA_TYPE)

        h = torch.cat(
            [
                atom_type_onehot,
                amino_idx_onehot,
                amino_types_onehot,
            ],
            dim=-1,
        )

        if self.num_valid_seq_lens > 1:
            seq_len_onehot = torch.nn.functional.one_hot(
                seq_len_encoding.expand(-1, num_atoms), num_classes=self.num_valid_seq_lens
            )
            h = torch.cat(
                [h, seq_len_onehot],
                dim=-1,
            )

        h = h.reshape(batch_size * num_atoms, -1).to(x.device) * node_mask
        h = torch.cat([h, t], dim=-1)
        h = h.reshape(batch_size * num_atoms, -1)

        edge_attr = torch.sum((x[edges[0]] - x[edges[1]]) ** 2, dim=1, keepdim=True)
        _, x_final = self.egnn(h, x, edges, edge_attr=edge_attr, node_mask=node_mask, edge_mask=edge_mask)
        vel = (x_final - x) * node_mask  # This masking operation is redundant but just in case

        vel = vel.view(batch_size, num_atoms, self.num_dimensions)
        vel = remove_mean_with_mask(vel, node_mask.view(batch_size, num_atoms, 1))

        self.counter += 1
        return vel.view(batch_size, num_atoms * self.num_dimensions)

    def get_adj_matrix(self, n_nodes, batch_size, device):
        if n_nodes in self.edges_dict:
            edges_dic_b = self.edges_dict[n_nodes]
            if batch_size in edges_dic_b:
                return edges_dic_b[batch_size]
            else:
                rows, cols = [], []
                for batch_idx in range(batch_size):
                    for i in range(n_nodes):
                        for j in range(n_nodes):
                            rows.append(i + batch_idx * n_nodes)
                            cols.append(j + batch_idx * n_nodes)
        else:
            self.edges_dict[n_nodes] = {}
            return self.get_adj_matrix(n_nodes, batch_size, device)

        edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]
        return edges
