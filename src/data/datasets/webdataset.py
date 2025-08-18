import logging

import numpy as np
import torch
import webdataset as wds
from huggingface_hub import list_repo_files


def build_webdataset(
    repo_id: str,
    repo_path: str,
    cache_dir: str | None = None,
    num_dimensions: int = 3,
    shuffle_buffer: int = 1000,
    num_aa_min: int | None = None,
    num_aa_max: int | None = None,
    transform=None,
):
    hf_url_prefix = f"https://huggingface.co/datasets/{repo_id}/resolve/main"
    all_hf_repo_files = list_repo_files(repo_id=repo_id, repo_type="dataset", revision="main")
    wds_hf_repo_files = [f for f in all_hf_repo_files if f.startswith(repo_path) and f.endswith(".tar")]
    wds_urls = [hf_url_prefix + "/" + f for f in wds_hf_repo_files]
    logging.info(f"Found {len(wds_urls)} WebDataset tar files in repo {repo_id} path {repo_path}")
    if cache_dir is not None:
        logging.info(f"Using WebDataset cache dir: {cache_dir}")

    def make_sample(sample):
        key, x = sample
        sequence = key.split("_")[0]
        x = np.frombuffer(x, dtype=np.float32).reshape(-1, num_dimensions)
        x = torch.from_numpy(x.copy())
        sample_dict = {"x": x, "sequence": sequence}
        if transform:
            sample_dict = transform(sample_dict)
        return sample_dict

    def seq_len_filter(sample):
        seq_len = len(sample[0].split("_")[0])
        if num_aa_min is not None and seq_len < num_aa_min:
            return False
        if num_aa_max is not None and seq_len > num_aa_max:
            return False
        return True

    dataset = (
        wds.WebDataset(
            wds_urls,
            cache_dir=cache_dir,
            shardshuffle=False,  # Ignored if resampled=True
            resampled=True,
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker,
        )
        .shuffle(shuffle_buffer)
        .to_tuple("__key__", "bin")
        .select(seq_len_filter)
        .map(make_sample)
    )
    return dataset
