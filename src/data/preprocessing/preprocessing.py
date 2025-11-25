import glob
import logging
import os
import pickle

import mdtraj as md
import openmm.app
from tqdm import tqdm

from src.data.preprocessing.encodings import get_encodings_dict
from src.data.preprocessing.permutations import get_permutations_dict

def prepare_preprocessing_cache(
    pdb_paths: list[str],
    cache_path: str,
    delimiter: str = ".",
) -> None:
    """
    Prepare and cache preprocessing data (PDB, topology, encodings, and permutations dicts).

    Checks if preprocessing cache file already exists. If not, prepares and caches
    all preprocessing dicts (PDB, topology, encodings, permutations) into a single pickle file.

    Args:
        pdb_paths: List of paths to PDB files to process.
        cache_path: Path to the cache pickle file.
        delimiter: Delimiter used to extract sequence names from PDB filenames. Defaults to ".".
    """
    if os.path.exists(cache_path):
        logging.info(f"Preprocessing cache already exists at {cache_path}")
        return
    
    logging.info(f"Preparing and caching preprocessing data for {len(pdb_paths)} files to {cache_path}")
    
    # Create PDB dict
    pdb_dict = {}
    logging.info("Loading PDBs...")
    for path in tqdm(pdb_paths, desc="Loading PDBs", total=len(pdb_paths)):
        pdb = openmm.app.PDBFile(path)
        assert len(list(pdb.topology.chains())) == 1, "Only single chain PDBs are supported"
        sequence = os.path.basename(path).split(delimiter)[0]
        pdb_dict[sequence] = pdb
    
    # Create topology dict
    logging.info("Extracting topologies...")
    topology_dict = {}
    for sequence, pdb in tqdm(pdb_dict.items(), desc="Extracting topologies"):
        topology = md.Topology.from_openmm(pdb.topology)
        topology_dict[sequence] = topology
    
    # Create encodings dict
    logging.info("Creating encodings...")
    encodings_dict = get_encodings_dict(topology_dict)
    
    # Create permutations dict
    logging.info("Creating permutations...")
    permutations_dict = get_permutations_dict(topology_dict)
    
    # Save all dicts to single cache file
    cache = {
        "pdb_dict": pdb_dict,
        "topology_dict": topology_dict,
        "encodings_dict": encodings_dict,
        "permutations_dict": permutations_dict,
    }
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    
    logging.info(f"Saved preprocessing cache to {cache_path}")


def load_preprocessing_cache(cache_path: str) -> tuple[dict, dict, dict, dict]:
    """
    Load preprocessing cache from a single pickle file.

    Args:
        cache_path: Path to the cache pickle file.

    Returns:
        Tuple of (pdb_dict, topology_dict, encodings_dict, permutations_dict).
    """
    with open(cache_path, "rb") as f:
        cache = pickle.load(f)  # noqa: S301
    
    return (
        cache["pdb_dict"],
        cache["topology_dict"],
        cache["encodings_dict"],
        cache["permutations_dict"],
    )