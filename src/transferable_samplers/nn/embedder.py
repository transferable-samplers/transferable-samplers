import math

import torch
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):
    def __init__(self, embed_size, div_value=10000):
        # max_len = 20 for long-term goal of oligopeptides
        super().__init__()
        self.embed_size = embed_size
        self.div_value = div_value

        assert self.embed_size % 2 == 0, "embed_size must be even."

    def forward(self, data):
        # TODO fix docstring
        position = data.float().unsqueeze(-1)  # Shape: [..., 1]
        div_term = torch.exp(
            torch.arange(0, self.embed_size, 2, device=data.device) * -(math.log(self.div_value) / self.embed_size)
        )  # Shape: [embed_size // 2]
        sinusoid_inp = position * div_term  # Shape: [..., embed_size // 2]
        pos_embedding = torch.zeros(*sinusoid_inp.shape[:-1], self.embed_size, device=data.device)
        pos_embedding[..., 0::2] = torch.sin(sinusoid_inp)
        pos_embedding[..., 1::2] = torch.cos(sinusoid_inp)
        return pos_embedding


class ConditionalEmbedder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        num_atom_emb: int = 128,
        num_residue_emb: int = 64,
        sinusoid_div_value: float = 0.0,
        embed_time: bool = False,
    ):
        """
        Input the value of the atom type, residue type, and residue position WITHOUT counting the padding token
        """

        super().__init__()

        self.atom_embed = nn.Embedding(num_embeddings=num_atom_emb, embedding_dim=hidden_dim)
        self.residue_embed = nn.Embedding(num_embeddings=num_residue_emb, embedding_dim=hidden_dim)
        self.residue_pos_embed = SinusoidalEmbedding(embed_size=hidden_dim, div_value=sinusoid_div_value)
        self.seq_len_embed = SinusoidalEmbedding(embed_size=hidden_dim, div_value=sinusoid_div_value)
        if embed_time:
            self.time_embed = SinusoidalEmbedding(embed_size=hidden_dim, div_value=10_000)  # always 10000 for time
        self.embed_time = embed_time
        self.mlp = nn.Sequential(
            nn.Linear(5 if embed_time else 4 * hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, atom_type, aa_type, aa_pos, seq_len, t=None, mask=None):
        if mask is None:
            mask = torch.ones_like(atom_type, dtype=torch.bool)

        if not ((t is None) ^ self.embed_time):
            raise ValueError("t must be provided if and only if embed_time is True")

        atom_emb = self.atom_embed(atom_type)
        residue_emb = self.residue_embed(aa_type)
        pos_embed = self.residue_pos_embed(aa_pos)

        # t / seq_len is of shape [b, 1], once embeded it will be [b, 1, channels]
        # so we expand it to [b, n, channels] to be concatenated with the other embeddings
        num_tokens = atom_type.shape[1]
        if self.embed_time:
            time_embed = self.time_embed(t).expand(-1, num_tokens, -1)
        seq_len_embed = self.seq_len_embed(seq_len).expand(-1, num_tokens, -1)

        x = torch.concat([atom_emb, residue_emb, pos_embed, seq_len_embed], dim=-1)
        if self.embed_time:
            x = torch.concat([x, time_embed], dim=-1)
        return self.mlp(x) * mask[..., None]  # [b, n, channels]


AtomConditionalEmbedder = ConditionalEmbedder
