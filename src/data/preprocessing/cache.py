import logging
import os
import pickle

import mdtraj as md
import openmm.app
from tqdm import tqdm

from src.data.preprocessing.encodings import get_encodings_dict
from src.data.preprocessing.permutations import get_permutations_dict


def prepare_and_cache_pdb_dict(
    pdb_paths: list[str], cache_path: str, delimiter: str = "-"
) -> dict[str, openmm.app.PDBFile]:
    if os.path.isfile(cache_path):
        logging.info(f"Loading cached PDB dict from {cache_path}")
        with open(cache_path, "rb") as f:
            pdb_dict = pickle.load(f)  # noqa:S301
    else:
        pdb_dict = {}
        logging.info(f"Creating pdb dict and caching to {cache_path}")
        for path in tqdm(pdb_paths, desc="Loading PDBs", total=len(pdb_paths)):
            pdb = openmm.app.PDBFile(path)
            assert len(list(pdb.topology.chains())) == 1, "Only single chain PDBs are supported"
            sequence = os.path.basename(path).split(delimiter)[0]
            pdb_dict[sequence] = pdb

        with open(cache_path, "wb") as f:
            pickle.dump(pdb_dict, f)

    return pdb_dict


def prepare_and_cache_topology_dict(pdb_dict: dict[str, openmm.app.PDBFile], cache_path: str) -> dict[str, md.Topology]:
    if os.path.isfile(cache_path):
        logging.info(f"Loading cached topology dict from {cache_path}")
        with open(cache_path, "rb") as f:
            topology_dict = pickle.load(f)  # noqa:S301
    else:
        logging.info(f"Creating topology dict and caching to {cache_path}")
        topology_dict = {}
        for sequence, pdb in tqdm(pdb_dict.items(), desc="Extracting topologies"):
            topology = md.Topology.from_openmm(pdb.topology)
            topology_dict[sequence] = topology

        with open(cache_path, "wb") as f:
            pickle.dump(topology_dict, f)

    return topology_dict


def prepare_and_cache_encodings_dict(topology_dict: dict[str, md.Topology], cache_path: str) -> dict[str, dict]:
    if os.path.isfile(cache_path):
        logging.info(f"Loading cached encodings dict from {cache_path}")
        with open(cache_path, "rb") as f:
            encodings_dict = pickle.load(f)  # noqa:S301
    else:
        logging.info(f"Creating encodings dict and caching to {cache_path}")
        encodings_dict = get_encodings_dict(topology_dict)
        with open(cache_path, "wb") as f:
            pickle.dump(encodings_dict, f)

    return encodings_dict


def prepare_and_cache_permutations_dict(topology_dict: dict[str, md.Topology], cache_path: str) -> dict[str, dict]:
    if os.path.isfile(cache_path):
        logging.info(f"Loading cached permutations dict from {cache_path}")
        with open(cache_path, "rb") as f:
            permutations_dict = pickle.load(f)  # noqa:S301
    else:
        logging.info(f"Creating permutations dict and caching to {cache_path}")
        permutations_dict = get_permutations_dict(topology_dict)
        with open(cache_path, "wb") as f:
            pickle.dump(permutations_dict, f)

    return permutations_dict
