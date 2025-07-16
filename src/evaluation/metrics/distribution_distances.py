import math
from typing import Union

import numpy as np
import ot as pot
import torch
from scipy.optimize import linear_sum_assignment

from .mmd import mix_rbf_mmd2
from .optimal_transport import wasserstein


def energy_distances(pred, true, prefix=""):
    pred = pred.cpu().numpy()
    true = true.cpu().numpy()
    energy_w2 = math.sqrt(pot.emd2_1d(true, pred))
    energy_w1 = pot.emd2_1d(true, pred, metric="euclidean")
    mean_dist = np.abs(pred.mean() - true.mean())
    cropped_pred = np.clip(pred, -1000, 1000)
    cropped_true = np.clip(true, -1000, 1000)
    cropped_energy_w2 = math.sqrt(pot.emd2_1d(cropped_true, cropped_pred))
    cropped_energy_w1 = pot.emd2_1d(cropped_true, cropped_pred, metric="euclidean")
    return_dict = {
        f"{prefix}/energy_w2": energy_w2,
        f"{prefix}/energy_w1": energy_w1,
        f"{prefix}/mean_dist": mean_dist,
        f"{prefix}/cropped_energy_w2": cropped_energy_w2,
        f"{prefix}/cropped_energy_w1": cropped_energy_w1,
    }
    return return_dict


def compute_distances(pred, true):
    """computes distances between vectors."""
    mse = torch.nn.functional.mse_loss(pred, true).item()
    me = math.sqrt(mse)
    mae = torch.mean(torch.abs(pred - true)).item()
    return mse, me, mae


def distribution_distances(pred: torch.Tensor, true: Union[torch.Tensor, list], prefix=""):
    """computes distances between distributions.
    pred: [batch, times, dims] tensor
    true: [batch, times, dims] tensor or list[batch[i], dims] of length times

    This handles jagged times as a list of tensors.
    """
    NAMES = [
        "1-Wasserstein",
        "2-Wasserstein",
        "RBF_MMD",
        "Mean_MSE",
        "Mean_L2",
        "Mean_L1",
        "Median_MSE",
        "Median_L2",
        "Median_L1",
        "Eq-EMD2",
    ]
    a = pred.cpu().view(pred.shape[0], -1)
    b = true.cpu().view(true.shape[0], -1)

    w1 = wasserstein(a, b, power=1)
    w2 = wasserstein(a, b, power=2)

    mmd_rbf = mix_rbf_mmd2(a, b, sigma_list=[0.01, 0.1, 1, 10, 100]).item()
    mean_dists = compute_distances(torch.mean(a, dim=0), torch.mean(b, dim=0))
    median_dists = compute_distances(torch.median(a, dim=0)[0], torch.median(b, dim=0)[0])
    dists = [w1, w2, mmd_rbf, *mean_dists, *median_dists]

    NAMES = [f"{prefix}/{name}" for name in NAMES]

    return dict(zip(NAMES, dists))


def find_rigid_alignment(A, B):
    """
    See: https://en.wikipedia.org/wiki/Kabsch_algorithm
    2-D or 3-D registration with known correspondences.
    Registration occurs in the zero centered coordinate system, and then
    must be transported back.
        Args:
        -    A: Torch tensor of shape (N,D) -- Point Cloud to Align (source)
        -    B: Torch tensor of shape (N,D) -- Reference Point Cloud (target)
        Returns:
        -    R: optimal rotation
        -    t: optimal translation
    Test on rotation + translation and on rotation + translation + reflection
        >>> A = torch.tensor([[1., 1.], [2., 2.], [1.5, 3.]], dtype=torch.float)
        >>> R0 = torch.tensor([[np.cos(60), -np.sin(60)], [np.sin(60), np.cos(60)]], dtype=torch.float)
        >>> B = (R0.mm(A.T)).T
        >>> t0 = torch.tensor([3., 3.])
        >>> B += t0
        >>> R, t = find_rigid_alignment(A, B)
        >>> A_aligned = (R.mm(A.T)).T + t
        >>> rmsd = torch.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
        >>> rmsd
        tensor(3.7064e-07)
        >>> B *= torch.tensor([-1., 1.])
        >>> R, t = find_rigid_alignment(A, B)
        >>> A_aligned = (R.mm(A.T)).T + t
        >>> rmsd = torch.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
        >>> rmsd
        tensor(3.7064e-07)
    """
    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = A_c.T.mm(B_c)
    U, S, V = torch.svd(H)
    # Rotation matrix
    R = V.mm(U.T)
    # Translation vector
    t = b_mean[None, :] - R.mm(a_mean[None, :].T).T
    t = t.T
    return R, t.squeeze()


def ot(x0, x1):
    dists = torch.cdist(x0, x1)
    _, col_ind = linear_sum_assignment(dists)
    x1 = x1[col_ind]
    return x1


def eot(x0, x1):
    M = []
    for i in range(len(x0)):
        reordered = []
        for j in range(len(x1)):
            x1_reordered = ot(x0[i], x1[j])
            reordered.append(x1_reordered)
        reordered = torch.stack(reordered)
        R, t = torch.vmap(find_rigid_alignment)(x0[i][None].repeat(len(x1), 1, 1), reordered)
        superimposed = torch.matmul(reordered, R)
        M.append(torch.cdist(x0[i].reshape(1, -1), superimposed.reshape(len(x1), -1)))
    M = torch.stack(M).squeeze()
    return pot.emd2(M=M, a=torch.ones(len(x0)) / len(x0), b=torch.ones(len(x1)) / len(x1))
