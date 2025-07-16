import mdtraj as md
import torch

from src.evaluation.metrics.optimal_transport import torus_wasserstein


def get_phi_psi_vectors(samples, topology):
    samples = samples.cpu()
    traj_samples = md.Trajectory(samples, topology=topology)
    phis = md.compute_phi(traj_samples)[1]
    psis = md.compute_psi(traj_samples)[1]
    return phis, psis


def ramachandran_metrics(true_samples, pred_samples, topology, prefix=""):
    phis_true, psis_true = get_phi_psi_vectors(true_samples, topology)
    x_true = torch.cat([torch.from_numpy(phis_true), torch.from_numpy(psis_true)], dim=1)

    phis_pred, psis_pred = get_phi_psi_vectors(pred_samples, topology)
    x_pred = torch.cat([torch.from_numpy(phis_pred), torch.from_numpy(psis_pred)], dim=1)

    metrics = {}
    metrics[prefix + "/torus_wasserstein"] = torus_wasserstein(x_true, x_pred)

    return metrics
