import numpy as np
import torch


def create_adjacency_list(distance_matrix, atom_types):
    adjacency_list = []

    # Iterate through the distance matrix
    num_nodes = len(distance_matrix)
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):  # Avoid duplicate pairs
            distance = distance_matrix[i][j]
            element_i = atom_types[i]
            element_j = atom_types[j]
            if 1 in (element_i, element_j):
                distance_cutoff = 0.14
            elif 4 in (element_i, element_j):
                distance_cutoff = 0.22
            elif 0 in (element_i, element_j):
                distance_cutoff = 0.18
            else:
                # elements should not be bonded
                distance_cutoff = 0.0

            # Add edge if distance is below the cutoff
            if distance < distance_cutoff:
                adjacency_list.append([i, j])

    return adjacency_list


def get_atom_types(topology):
    atom_dict = {"C": 0, "H": 1, "N": 2, "O": 3, "S": 4}
    atom_types = []
    for atom_name in topology.atoms:
        atom_types.append(atom_name.name[0])
    atom_types = torch.from_numpy(np.array([atom_dict[atom_type] for atom_type in atom_types]))

    return atom_types


def get_adj_list(topology):
    adj_list = torch.from_numpy(
        np.array(
            [(b.atom1.index, b.atom2.index) for b in topology.bonds],
            dtype=np.int32,
        )
    )
    return adj_list


def find_chirality_centers(adj_list: torch.Tensor, atom_types: torch.Tensor, num_h_atoms: int = 2) -> torch.Tensor:
    """
    Return the chirality centers for a peptide, e.g. carbon alpha atoms and their bonds.

    Args:
        adj_list: List of bonds
        atom_types: List of atom types
        num_h_atoms: If num_h_atoms or more hydrogen atoms connected to the center, it is not reportet.
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


def compute_chirality_sign(coords: torch.Tensor, chirality_centers: torch.Tensor) -> torch.Tensor:
    """
    Compute indicator signs for a given configuration.
    If the signs for two configurations are different for the same center, the chirality changed.

    Args:
        coords: Tensor of atom coordinates
        chirality_centers: List of chirality_centers

    Returns:
        Indicator signs
    """
    assert coords.dim() == 3
    # print(coords.shape, chirality_centers.shape, chirality_centers)
    direction_vectors = coords[:, chirality_centers[:, 1:], :] - coords[:, chirality_centers[:, [0]], :]
    perm_sign = torch.einsum(
        "ijk, ijk->ij",
        direction_vectors[:, :, 0],
        torch.cross(direction_vectors[:, :, 1], direction_vectors[:, :, 2], dim=-1),
    )
    return torch.sign(perm_sign)


def check_symmetry_change(true_coords: torch.Tensor, pred_coords: torch.Tensor, adj_list, atom_types) -> torch.Tensor:
    """
    Check for a batch if the chirality changed wrt to some reference reference_signs.
    If the signs for two configurations are different for the same center, the chirality changed.

    Args:
        true_coords: Tensor of atom coordinates
        pred_coords: Tensor of atom coordinates
        TODO
    Returns:
        Mask, where changes are True
    """
    chirality_centers = find_chirality_centers(adj_list, atom_types)

    reference_signs = compute_chirality_sign(true_coords[[1]], chirality_centers)
    perm_sign = compute_chirality_sign(pred_coords, chirality_centers)
    return (perm_sign != reference_signs.to(pred_coords)).any(dim=-1)


def get_symmetry_change(true_samples, pred_samples, topology):
    true_samples = true_samples[: len(pred_samples)]

    adj_list = get_adj_list(topology)
    atom_types = get_atom_types(topology)

    symmetry_change = check_symmetry_change(true_samples, pred_samples, adj_list, atom_types)
    return symmetry_change

def fix_chirality(true_samples, samples_data, topology, drop_unfixable: bool = False):
    """
    Fix chirality issues in predicted samples by comparing to true samples.
    
    Args:
        true_samples: True reference samples (normalized)
        samples_data: SamplesData object containing predicted samples
        topology: Topology object for the peptide
        drop_unfixable: Whether to drop samples that can't be fixed
        
    Returns:
        SamplesData: Fixed samples data
    """
    from src.utils.dataclasses import SamplesData
    
    pred_samples = samples_data.x.clone()
    temp_pred_samples = pred_samples.clone()

    first_symmetry_change = get_symmetry_change(
        true_samples,
        temp_pred_samples,
        topology,
    )

    correct_symmetry_rate = 1 - first_symmetry_change.float().mean().item()

    temp_pred_samples[first_symmetry_change] *= -1

    second_symmetry_change = get_symmetry_change(
        true_samples,
        temp_pred_samples,
        topology,
    )

    uncorrectable_symmetry_rate = second_symmetry_change.float().mean().item()

    # Fix the symmetry
    pred_samples[first_symmetry_change] *= -1

    if drop_unfixable:  # only makes sense to drop if symmetry is fixed
        mask = ~second_symmetry_change
        pred_samples = pred_samples[mask]
        if samples_data.proposal_energy is not None:
            proposal_energy = samples_data.proposal_energy[mask]
        else:
            proposal_energy = None
        target_energy = samples_data.target_energy[mask]
        if samples_data.importance_logits is not None:
            importance_logits = samples_data.importance_logits[mask]
        else:
            importance_logits = None
    else:
        proposal_energy = samples_data.proposal_energy
        target_energy = samples_data.target_energy
        importance_logits = samples_data.importance_logits

    return SamplesData(
        x=pred_samples,
        proposal_energy=proposal_energy,
        target_energy=target_energy,
        importance_logits=importance_logits,
    )