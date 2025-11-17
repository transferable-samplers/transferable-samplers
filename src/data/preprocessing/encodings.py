import torch
from tqdm import tqdm

"""Start encodings from 1 to leave 0 for zero padding"""

ATOM_TYPE_encodings_DICT = {
    "C": 1,
    "CA": 2,
    "CB": 3,
    "CD": 4,
    "CD1": 5,
    "CD2": 6,
    "CE": 7,
    "CE1": 8,
    "CE2": 9,
    "CE3": 10,
    "CG": 11,
    "CG1": 12,
    "CG2": 13,
    "CH2": 14,
    "CZ": 15,
    "CZ2": 16,
    "CZ3": 17,
    "H": 18,
    "HA": 19,
    "HB": 20,
    "HD": 21,
    "HD1": 22,
    "HD2": 23,
    "HE": 24,
    "HE1": 25,
    "HE2": 26,
    "HE3": 27,
    "HG": 28,
    "HG1": 29,
    "HG2": 30,
    "HH": 31,
    "HH1": 32,
    "HH2": 33,
    "HZ": 34,
    "HZ2": 35,
    "HZ3": 36,
    "N": 37,
    "ND1": 38,
    "ND2": 39,
    "NE": 40,
    "NE1": 41,
    "NE2": 42,
    "NH1": 43,
    "NH2": 44,
    "NZ": 45,
    "O": 46,
    "OD": 47,
    "OE": 48,
    "OG": 49,
    "OG1": 50,
    "OH": 51,
    "OXT": 52,
    "SD": 53,
    "SG": 54,
}

AA_TYPE_encodings_DICT = {
    "ALA": 1,
    "ARG": 2,
    "ASN": 3,
    "ASP": 4,
    "CYS": 5,
    "GLN": 6,
    "GLU": 7,
    "GLY": 8,
    "HIS": 9,
    "ILE": 10,
    "LEU": 11,
    "LYS": 12,
    "MET": 13,
    "PHE": 14,
    "PRO": 15,
    "SER": 16,
    "THR": 17,
    "TRP": 18,
    "TYR": 19,
    "VAL": 20,
}

AA_CODE_CONVERSION = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def get_encodings(topology):
    aa_pos_encodings = []
    aa_type_encodings = []
    atom_type_encodings = []

    for i, aa in enumerate(topology.residues):
        for atom in aa.atoms:
            aa_pos_encodings.append(i + 1)  # shifted to account for pad tokens
            aa_type_encodings.append(AA_TYPE_encodings_DICT[aa.name])

            atom_name = atom.name

            # TODO double check this with Leon
            # Standardize side-chain H atom encodings
            if atom_name[0] == "H" and atom_name[-1] in ("1", "2", "3"):
                # For these AA the H-X-N atoms are not interchangeable
                if aa.name in ("HIS", "HIE", "PHE", "TRP", "TYR") and atom_name[:2] in (
                    "HE",
                    "HD",
                    "HZ",
                    "HH",
                ):
                    pass
                else:
                    atom_name = atom_name[:-1]

            # Standardize side-chain O atom encodings
            if atom_name[:2] == "OE" or atom_name[:2] == "OD":
                atom_name = atom_name[:-1]

            atom_type_encodings.append(ATOM_TYPE_encodings_DICT[atom_name])

    atom_type_encodings = torch.tensor(atom_type_encodings, dtype=torch.int64)
    aa_pos_encodings = torch.tensor(aa_pos_encodings, dtype=torch.int64)
    aa_type_encodings = torch.tensor(aa_type_encodings, dtype=torch.int64)

    encodings = {
        "atom_type": atom_type_encodings,
        "aa_pos": aa_pos_encodings,
        "aa_type": aa_type_encodings,
        "seq_len": torch.tensor([topology.n_residues], dtype=torch.int64),
    }

    return encodings


def get_encodings_dict(topology_dict):
    encodings_dict = {}

    for i, (sequence, topology) in tqdm(
        enumerate(topology_dict.items()),
        desc="Generating encodings",
        total=len(topology_dict),
    ):
        encodings_dict[sequence] = get_encodings(topology)

    return encodings_dict
