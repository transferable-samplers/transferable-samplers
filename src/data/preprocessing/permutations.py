from collections import defaultdict
from itertools import product
from pathlib import Path

import torch
import yaml
from tqdm import tqdm


# Load a YAML file and return its contents as a dictionary
def load_yaml_as_dict(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r") as f:
        return yaml.safe_load(f)


def standardize_atom_name(atom_name: str, aa_name: str) -> str:
    if atom_name[0] == "H" and atom_name[-1] in ("1", "2", "3"):
        # For these AA the H-X-N atoms are not interchangable
        if aa_name in ("HIS", "HIE", "PHE", "TRP", "TYR") and atom_name[:2] in (
            "HE",
            "HD",
            "HZ",
            "HH",
        ):
            pass
        else:
            atom_name = atom_name[:-1]

    # Standarize side-chain O atom encodings
    if atom_name[:2] == "OE" or atom_name[:2] == "OD":
        atom_name = atom_name[:-1]

    return atom_name


def get_permutation(
    permutations_definition_dict, topology, sequence_ordering, global_type, sidechain_variant, residue_cache=None
):
    # Validate input strategy options
    if sequence_ordering not in ["n2c", "c2n"]:
        raise ValueError(f"Unknown sequence ordering: {sequence_ordering}")
    if global_type not in ["residue-by-residue", "backbone-first"]:
        raise ValueError(f"Unknown global type: {global_type}")
    if sidechain_variant not in ["standard", "variant"]:
        raise ValueError(f"Unknown sidechain variant: {sidechain_variant}")

    # Get residues in forward or reverse order
    residue_list = list(topology.residues)
    if sequence_ordering != "n2c":
        residue_list = list(reversed(residue_list))

    # Backbone atom permutations for this ordering (e.g., N2C or C2N)
    before_sidechain_permutation_definition = permutations_definition_dict["backbone"][sequence_ordering][
        "before_sidechain"
    ]
    after_sidechain_permutation_definition = permutations_definition_dict["backbone"][sequence_ordering][
        "after_sidechain"
    ]

    # List to store per-residue permutations split into before sidechain, after sidechain, and sidechain
    before_sidechain_permutations = []
    after_sidechain_permutations = []
    sidechain_permutations = []

    # Iterate through each residue to build permutations
    for i, residue in enumerate(residue_list):
        # Get standarized residue name with terminal labels for cache
        residue_name = residue.name if residue.name != "HIS" else "HIE"  # Timewarp data has HIE labelled as HIS
        assert residue_name in permutations_definition_dict["sidechain"], (
            f"Residue {residue_name} not found in sidechain definitions"
        )
        if (i == 0 and sequence_ordering == "n2c") or (i == len(residue_list) - 1 and sequence_ordering == "c2n"):
            residue_name_with_terminals = "N-" + residue_name
        elif (i == 0 and sequence_ordering == "c2n") or (i == len(residue_list) - 1 and sequence_ordering == "n2c"):
            residue_name_with_terminals = "C-" + residue_name
        else:
            residue_name_with_terminals = residue_name

        # Check if the residue has a cache entry
        input_atom_ordering = [standardize_atom_name(atom.name, residue_name) for atom in list(residue.atoms)]
        first_atom_index = list(residue.atoms)[0].index
        if residue_cache is not None and residue_cache.get(residue_name_with_terminals) is not None:
            cached = residue_cache[residue_name_with_terminals]
            if input_atom_ordering != cached["input_atom_ordering"]:
                # Assertion to check consistent residue definition
                raise AssertionError(
                    f"Inconsistent atom name order for {residue_name_with_terminals}.\n"
                    f"Expected: {cached['input_atom_ordering']}\nFound:    {input_atom_ordering}"
                )
            # Add to per-residue list from cache
            before_sidechain_permutations.append(cached["before_sidechain"] + first_atom_index)
            after_sidechain_permutations.append(cached["after_sidechain"] + first_atom_index)
            sidechain_permutations.append(cached["sidechain"] + first_atom_index)
        else:
            # Get permutation rules for the residue's sidechain
            sidechain_permutations_definition_dict = permutations_definition_dict["sidechain"][residue_name]

            # Choose the appropriate variant strategy for sidechain atom ordering
            if sidechain_variant == "variant":
                if "ring_reverse" in sidechain_permutations_definition_dict:
                    sidechain_permutation_definition = sidechain_permutations_definition_dict["ring_reverse"]
                elif "branch_order_reverse" in sidechain_permutations_definition_dict:
                    sidechain_permutation_definition = sidechain_permutations_definition_dict["branch_order_reverse"]
                else:
                    sidechain_permutation_definition = sidechain_permutations_definition_dict["standard"]
            else:
                sidechain_permutation_definition = sidechain_permutations_definition_dict["standard"]

            # Check for overlap between backbone and sidechain definitions
            overlap = set(before_sidechain_permutation_definition) & set(sidechain_permutation_definition)
            if overlap:
                raise ValueError(f"Atom(s) {overlap} defined in both backbone and sidechain for residue {residue_name}")
            overlap = set(after_sidechain_permutation_definition) & set(sidechain_permutation_definition)
            if overlap:
                raise ValueError(f"Atom(s) {overlap} defined in both backbone and sidechain for residue {residue_name}")

            # Build name → atom index map once
            atom_index_by_name = {atom.name: atom.index for atom in residue.atoms}

            # Enforce ordering based on YAML definition
            residue_before_sidechain_permutation = [
                atom_index_by_name[name]
                for name in before_sidechain_permutation_definition
                if name in atom_index_by_name  # Backbone atoms may not be present in all residues
            ]
            residue_after_sidechain_permutation = [
                atom_index_by_name[name]
                for name in after_sidechain_permutation_definition
                if name in atom_index_by_name  # Backbone atoms may not be present in all residues
            ]
            residue_sidechain_permutation = [atom_index_by_name[name] for name in sidechain_permutation_definition]

            residue_before_sidechain_permutation = torch.tensor(residue_before_sidechain_permutation, dtype=torch.long)
            residue_after_sidechain_permutation = torch.tensor(residue_after_sidechain_permutation, dtype=torch.long)
            residue_sidechain_permutation = torch.tensor(residue_sidechain_permutation, dtype=torch.long)

            # Save the permutation per residue
            before_sidechain_permutations.append(residue_before_sidechain_permutation)
            after_sidechain_permutations.append(residue_after_sidechain_permutation)
            sidechain_permutations.append(residue_sidechain_permutation)

            if residue_cache is not None:
                # Cache the backbone and sidechain permutations for this residue, as well as the atom names in order
                residue_cache[residue_name_with_terminals] = {
                    "input_atom_ordering": input_atom_ordering,
                    "before_sidechain": residue_before_sidechain_permutation - first_atom_index,
                    "after_sidechain": residue_after_sidechain_permutation - first_atom_index,
                    "sidechain": residue_sidechain_permutation - first_atom_index,
                }

    # Flatten the residue-wise permutations into a full permutation
    permutation = []
    if global_type == "residue-by-residue":
        for i in range(topology.n_residues):  # Loop through residues once
            permutation.append(before_sidechain_permutations[i])
            permutation.append(sidechain_permutations[i])
            permutation.append(after_sidechain_permutations[i])
    elif global_type == "backbone-first":
        # find indices corresponding to hydrogens
        hydrogen_indices = set(
            atom.index for residue in residue_list for atom in residue.atoms if atom.name.startswith("H")
        )
        before_sidechain_hydrogen_permutations = []
        after_sidechain_hydrogen_permutations = []
        for i in range(topology.n_residues):  # Loop through residues once - taking backbone atoms
            # add the heavy backbone atoms directly to the permutation
            permutation.append(
                torch.tensor(
                    [
                        atom_index
                        for atom_index in [*before_sidechain_permutations[i], *after_sidechain_permutations[i]]
                        if atom_index.item() not in hydrogen_indices
                    ],
                    dtype=torch.long,
                )
            )
            # store the hydrogen indices separately
            before_sidechain_hydrogen_permutations.append(
                torch.tensor(
                    [
                        atom_index
                        for atom_index in before_sidechain_permutations[i]
                        if atom_index.item() in hydrogen_indices
                    ],
                    dtype=torch.long,
                )
            )
            after_sidechain_hydrogen_permutations.append(
                torch.tensor(
                    [
                        atom_index
                        for atom_index in after_sidechain_permutations[i]
                        if atom_index.item() in hydrogen_indices
                    ],
                    dtype=torch.long,
                )
            )
        for i in range(
            topology.n_residues
        ):  # Loop through residues again - taking backbone hydrogens and sidechain atoms
            # add backbone hydrogens + sidechain atoms
            permutation.append(before_sidechain_hydrogen_permutations[i])
            permutation.append(sidechain_permutations[i])
            permutation.append(after_sidechain_hydrogen_permutations[i])

    permutation = torch.cat(permutation)

    # Final integrity checks
    assert len(permutation) == topology.n_atoms, (
        f"Permutation length (must be {topology.n_atoms}, got {len(permutation)})"
    )
    assert max(permutation) == topology.n_atoms - 1, (
        f"Permutation max index (must be {topology.n_atoms - 1}, got {max(permutation)})"
    )
    assert min(permutation) == 0, f"Permutation min index (must be 0, got {min(permutation)})"
    unique, counts = permutation.unique(return_counts=True)
    duplicates = unique[counts > 1]
    assert len(duplicates) == 0, f"Permutation contains duplicate atom indices: {duplicates.tolist()}"

    return permutation


def get_permutations_dict(topology_dict):
    # Load the permutations definition from YAML file
    permutations_definition_dict = load_yaml_as_dict("src/data/preprocessing/permutations.yaml")

    permutations_dict = defaultdict(dict)

    sequence_orderings = ["n2c", "c2n"]
    global_types = ["residue-by-residue", "backbone-first"]
    sidechain_variants = ["standard", "variant"]

    # Generate all combinations of configuration settings
    configs = list(product(sequence_orderings, global_types, sidechain_variants))
    total = len(configs) * len(topology_dict)  # total number of permutations to generate

    with tqdm(total=total, desc="Generating permutations") as pbar:  # progress bar for tracking progress
        for sequence_ordering, global_type, sidechain_variant in configs:
            residue_cache = {}  # New cache for each configuration
            for sequence, topology in topology_dict.items():
                key = f"{sequence_ordering}_{global_type}_{sidechain_variant}"
                permutation = get_permutation(
                    permutations_definition_dict,
                    topology,
                    sequence_ordering,
                    global_type,
                    sidechain_variant,
                    residue_cache,
                )
                permutations_dict[sequence][key] = permutation
                permutations_dict[sequence][key + "_flip"] = torch.flip(
                    permutation, dims=[0]
                )  # Also add flipped version of the permutation

                pbar.update(1)

    return permutations_dict
