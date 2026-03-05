"""Handling of chirality checks for peptides."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _get_atom_types(topology: Any) -> torch.Tensor:
    atom_dict = {"C": 0, "H": 1, "N": 2, "O": 3, "S": 4}
    atom_types = []
    for atom_name in topology.atoms:
        atom_types.append(atom_name.name[0])
    atom_types = torch.from_numpy(np.array([atom_dict[atom_type] for atom_type in atom_types]))

    return atom_types


def _get_adj_list(topology: Any) -> torch.Tensor:
    adj_list = torch.from_numpy(
        np.array(
            [(b.atom1.index, b.atom2.index) for b in topology.bonds],
            dtype=np.int32,
        )
    )
    return adj_list


def _find_chirality_centers(adj_list: torch.Tensor, atom_types: torch.Tensor, num_h_atoms: int = 2) -> torch.Tensor:
    """Return the chirality centers for a peptide, e.g. carbon alpha atoms and their bonds.

    Args:
        adj_list: List of bonds
        atom_types: List of atom types
        num_h_atoms: If num_h_atoms or more hydrogen atoms connected to the center, it is not reported.
            Default is 2, because in this case the mirroring is a simple permutation.

    Returns:
        chirality_centers
    """
    chirality_centers = []
    candidate_chirality_centers = torch.where(torch.unique(adj_list, return_counts=True)[1] == 4)[0]
    for center in candidate_chirality_centers:
        bond_idx, bond_pos = torch.where(adj_list == center)
        bonded_idxs = adj_list[bond_idx, (bond_pos + 1) % 2].long()
        adj_types = atom_types[bonded_idxs]
        if torch.count_nonzero(adj_types - 1) > num_h_atoms:
            chirality_centers.append([center, *bonded_idxs[:3]])
    return torch.tensor(chirality_centers).to(adj_list).long()


def _compute_chirality_sign(coords: torch.Tensor, chirality_centers: torch.Tensor) -> torch.Tensor:
    """Compute indicator signs for a given configuration.

    If the signs for two configurations are different for the same center, the chirality changed.

    Args:
        coords: Tensor of atom coordinates
        chirality_centers: List of chirality_centers

    Returns:
        Indicator signs
    """
    assert coords.dim() == 3
    direction_vectors = coords[:, chirality_centers[:, 1:], :] - coords[:, chirality_centers[:, [0]], :]
    perm_sign = torch.einsum(
        "ijk, ijk->ij",
        direction_vectors[:, :, 0],
        torch.cross(direction_vectors[:, :, 1], direction_vectors[:, :, 2], dim=-1),
    )
    return torch.sign(perm_sign)


def _check_symmetry_change(
    true_coords: torch.Tensor,
    pred_coords: torch.Tensor,
    adj_list: torch.Tensor,
    atom_types: torch.Tensor,
) -> torch.Tensor:
    """Check if the chirality changed wrt to some reference reference_signs.

    If the signs for two configurations are different for the same center, the chirality changed.

    Args:
        true_coords: Tensor of atom coordinates.
        pred_coords: Tensor of atom coordinates.
        adj_list: Bond adjacency list.
        atom_types: Atom type indices.

    Returns:
        Mask, where ``True`` indicates a chirality change.
    """
    chirality_centers = _find_chirality_centers(adj_list, atom_types)

    reference_signs = _compute_chirality_sign(true_coords[[1]], chirality_centers)
    perm_sign = _compute_chirality_sign(pred_coords, chirality_centers)
    return (perm_sign != reference_signs.to(pred_coords)).any(dim=-1)


def get_symmetry_change(true_samples: torch.Tensor, pred_samples: torch.Tensor, topology: md.Topology) -> torch.Tensor:
    """Check whether predicted samples have inconsistent global chirality relative to true samples.

    Args:
        true_samples: Reference atom coordinates, shape ``(N, num_atoms, 3)``.
        pred_samples: Predicted atom coordinates, shape ``(M, num_atoms, 3)``.
        topology: Molecular topology (e.g. from MDTraj) providing atoms and bonds.

    Returns:
        Boolean mask of shape ``(M,)`` where ``True`` indicates a chirality flip.
    """
    true_samples = true_samples[: len(pred_samples)]

    adj_list = _get_adj_list(topology)
    atom_types = _get_atom_types(topology)

    symmetry_change = _check_symmetry_change(true_samples, pred_samples, adj_list, atom_types)
    return symmetry_change
