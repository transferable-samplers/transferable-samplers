import logging
import os
import tarfile

from huggingface_hub import hf_hub_download, snapshot_download

# TODO repos currently hardcoded - slightly hard to remove hardcode from data as assumes repo structure


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

    REPO_ID = "transferable-samplers/many-peptides-md"

    logging.info(f"Downloading and extracting PDB tarfiles to {data_dir}")

    # Snapshot download automatically avoids re-downloading if already present
    local_path = snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=data_dir,
        allow_patterns=["pdb_tarfiles/*"],
        use_auth_token=True,
    )

    for subset in ["train", "val", "test"]:
        tar_filepath = local_path + f"/pdb_tarfiles/{subset}.tar"
        if not os.path.exists(tar_filepath):
            continue
        safe_extract_tar(tar_filepath, os.path.join(data_dir, "pdbs"))


def download_evaluation_data(data_dir: str):
    """
    Downloads and extracts the evaluation trajectories_subsampled from Hugging Face Hub.

    Args:
        data_dir (str): The top-level data dir in which to build the evaluation data subdirectories
    """

    REPO_ID = "transferable-samplers/many-peptides-md"

    logging.info(f"Downloading evaluation data to {data_dir}")

    # Snapshot download automatically avoids re-downloading if already present
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=data_dir,
        allow_patterns=["trajectories_subsampled/*"],
        use_auth_token=True,
    )
