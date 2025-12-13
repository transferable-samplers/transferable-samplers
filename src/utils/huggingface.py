import logging
import os
import tarfile

from huggingface_hub import hf_hub_download, snapshot_download

# TODO repos currently hardcoded - slightly hard to remove hardcode from data as assumes repo structure
REPO_ID = "transferable-samplers/many-peptides-md"


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
            revision="fixing-tica",  # TODO once HF merge complete
            allow_patterns=["trajectories_subsampled/*"],
            use_auth_token=True,
        )
    else:
        logging.info(f"Evaluation data already present in {data_dir}")
