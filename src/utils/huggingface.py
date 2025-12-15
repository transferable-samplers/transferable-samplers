import logging
import os
import tarfile

import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download

# TODO repos currently hardcoded - slightly hard to remove hardcode from data as assumes repo structure
REPO_ID = "transferable-samplers/many-peptides-md"

TICA_MEAN_SHAPES = {
    "test": {
        "AA": 34,
        "CE": 61,
        "CL": 72,
        "DG": 34,
        "DI": 72,
        "FK": 159,
        "HL": 126,
        "HM": 126,
        "IK": 111,
        "IM": 97,
        "KG": 61,
        "LE": 84,
        "MQ": 97,
        "NA": 51,
        "NC": 61,
        "PG": 42,
        "PY": 126,
        "QR": 142,
        "RL": 142,
        "RT": 111,
        "SS": 34,
        "TD": 51,
        "VF": 126,
        "VS": 51,
        "WA": 142,
        "WH": 237,
        "WQ": 196,
        "WS": 142,
        "YC": 111,
        "YQ": 142,
        "ARIP": 369,
        "CCVH": 318,
        "CIPQ": 318,
        "DEMT": 271,
        "DMTL": 294,
        "EHQW": 613,
        "FESD": 318,
        "FYYY": 798,
        "GCDE": 189,
        "GDTI": 208,
        "GGRS": 208,
        "HEAV": 318,
        "HQVS": 343,
        "HYGW": 613,
        "ITYL": 424,
        "KKAP": 343,
        "KLLR": 514,
        "KRWN": 684,
        "NCFG": 294,
        "NEVI": 318,
        "PQIF": 453,
        "QAKR": 424,
        "QWNL": 546,
        "RLMM": 483,
        "SHKS": 318,
        "SVND": 228,
        "TAPF": 318,
        "TMWC": 453,
        "VPFY": 514,
        "WNMA": 453,
        "ANKSMIEA": 1077,
        "CGSWHKQR": 1753,
        "CLCCGQWN": 1317,
        "DDRDTEQT": 1170,
        "DGVAHALS": 903,
        "EKYYWMQT": 2187,
        "FWRVDHDM": 2122,
        "GNDLVTVI": 1032,
        "HWHSLICK": 1933,
        "IDHRQLKW": 2187,
        "IFGWVYTG": 1638,
        "ISKCKNGE": 1123,
        "KRRGFFLE": 2058,
        "MAPQTIAT": 1032,
        "MRDPVLFA": 1527,
        "MWNSTEMI": 1527,
        "MYGRNCYM": 1695,
        "NHQYGSDP": 1267,
        "NKEKFFQH": 2058,
        "NPCLCYML": 1420,
        "PGESTAES": 745,
        "PLFHVMYV": 1872,
        "PPWRECNN": 1695,
        "PYIRNCVE": 1582,
        "SPHKMRLC": 1582,
        "SQQKVAFE": 1368,
        "VWIPVIDT": 1527,
        "WDLIQFRQ": 2187,
        "WTYAFAHS": 1753,
        "YFPHAGYT": 1638,
    },
    "val": {
        "AL": 61,
        "EC": 61,
        "FT": 111,
        "GQ": 51,
        "HV": 111,
        "ND": 61,
        "PF": 126,
        "RS": 97,
        "WI": 196,
        "WY": 259,
        "CAVQ": 249,
        "CVNR": 369,
        "EDWY": 579,
        "GNFL": 343,
        "ISDT": 228,
        "KHAP": 369,
        "KHGM": 369,
        "LFTE": 396,
        "PMWS": 453,
        "YRIQ": 579,
        "AHKILWHG": 1812,
        "AMYNLMIR": 1695,
        "DQNVCCME": 1123,
        "IVTVAYFP": 1473,
        "PYTQVRPC": 1527,
        "QDFKRNIR": 1995,
        "QTHSSDEF": 1267,
        "SMFNWLKY": 2122,
        "TGEAGKSH": 903,
        "WCEPGLDW": 1695,
    },
}


def download_weights(destination_dir: str, hf_filepath: str) -> str:
    """
    Downloads the model weights from Hugging Face Hub.

    Args:
        destination_dir (str): The destination dir where the model weights will be saved.
        hf_filepath (str): The filepath in the Hugging Face repo
    """

    REPO_ID = "transferable-samplers/model-weights"

    try:
        os.makedirs(destination_dir, exist_ok=True)
        local_path = hf_hub_download(repo_id=REPO_ID, filename=hf_filepath, local_dir=destination_dir)
        print(f"Model weights downloaded successfully to {destination_dir}")
        return local_path
    except Exception as e:
        print(f"Failed to download model weights: {e}")


def safe_extract_tar(tar_path, extraction_path):
    """
    Safely extracts a tar archive to the given path.

    Args:
        tar_path (str): Path to the tar archive.
        extraction_path (str): Directory to extract files into.
    """
    abs_path = os.path.abspath(extraction_path)

    with tarfile.open(tar_path) as tar:
        for member in tar.getmembers():
            member_path = os.path.abspath(os.path.join(extraction_path, member.name))
            if not member_path.startswith(abs_path + os.sep):
                raise Exception(f"Blocked path traversal attempt: {member.name}")
        tar.extractall(extraction_path)  # noqa: S202 - safe, paths validated above


def download_and_extract_pdb_tarfiles(data_dir: str):
    """
    Downloads and extracts the PDB tarfiles from Hugging Face Hub.

    Args:
        data_dir (str): The top-level data dir in which to build the pdb file subdirectories
    """
    if not os.path.exists(os.path.join(data_dir, "pdb_tarfiles")):
        logging.info(f"Downloading PDB tarfiles to {data_dir}")
        # Snapshot download automatically avoids re-downloading if already present
        local_path = snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=data_dir,
            allow_patterns=["pdb_tarfiles/*"],
            use_auth_token=True,
        )
        logging.info(f"Extracting PDB tarfiles to {data_dir}")
        for subset in ["train", "val", "test"]:
            tar_filepath = local_path + f"/pdb_tarfiles/{subset}.tar"
            safe_extract_tar(tar_filepath, os.path.join(data_dir, "pdbs"))
    else:
        logging.info(f"PDB tarfiles already present in {data_dir}")


def download_evaluation_data(data_dir: str):
    """
    Downloads and extracts the evaluation trajectories_subsampled from Hugging Face Hub.

    Args:
        data_dir (str): The top-level data dir in which to build the evaluation data subdirectories
    """

    logging.warning("Critical Update")
    logging.warning(
        "The original 8AA TICA models within `subsampled_trajectories/*/8AA/*.npz` employed a CA-only atom selection."
    )
    logging.warning("These models are not valid for comparison to results in our paper.")
    logging.warning("Updated files (uploaded [DATE])** now contain corrected models.")
    logging.warning("If you previously downloaded this dataset, please re-download to ensure accurate results.")
    logging.warning("Note: Codebase references to `tica_features_ca` must now be replaced with `tica_features`.")
    logging.warning("This was resolved in our codebase by PR #N.")
    logging.warning(
        "Note: Unguarded `snapshot_download` calls will automatically redownload the relevant files "
        "when it detects a change in the repo."
    )
    logging.warning("We sincerely apologize for any inconvenience this may have caused.")
    if not os.path.exists(os.path.join(data_dir, "trajectories_subsampled")):
        logging.info(f"Downloading evaluation data to {data_dir}")
        # Snapshot download automatically avoids re-downloading if already present
        snapshot_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            local_dir=data_dir,
            allow_patterns=["trajectories_subsampled/*"],
            use_auth_token=True,
        )
    else:
        logging.info(f"Evaluation data already present in {data_dir}")

    # Check TICA shapes for all downloaded files
    logging.info("Checking TICA model shapes...")

    for subset_key in TICA_MEAN_SHAPES.keys():
        for sequence in TICA_MEAN_SHAPES[subset_key].keys():
            # Construct file path from dict keys
            file_path = os.path.join(
                data_dir, "trajectories_subsampled", subset_key, f"{len(sequence)}AA", f"{sequence}_subsampled.npz"
            )
            data = np.load(file_path)
            tica_mean = data["tica_mean"]
            tica_mean_shape = tica_mean.shape[0]
            expected_shape = TICA_MEAN_SHAPES[subset_key][sequence]

            error_message = (
                f"TICA mean shape for {sequence} is {tica_mean_shape}, but expected {expected_shape}\n"
                f"File: {file_path}\n"
                "This is likely due to the change in the TICA model implementation.\n"
                "See https://huggingface.co/datasets/transferable-samplers/many-peptides-md for further details."
            )

            assert tica_mean_shape == expected_shape, error_message

    logging.info("TICA model shape checks completed successfully.")
