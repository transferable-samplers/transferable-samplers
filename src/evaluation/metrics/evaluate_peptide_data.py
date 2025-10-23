import logging

from src.evaluation.metrics.distribution_distances import (
    distribution_distances,
    energy_distances,
)
from src.evaluation.metrics.ess import sampling_efficiency
from src.evaluation.metrics.jsd_metric import jsd_metric, jsd_torus_metric
from src.evaluation.metrics.ramachandran import ramachandran_metrics
from src.evaluation.metrics.tica_metric import tica_metric


def evaluate_peptide_data(
    true_data,
    pred_data,
    topology,
    tica_model,
    num_eval_samples=None,
    compute_distribution_distances: bool = True,
    prefix: str = "",
):
    """Computes all metrics between true and predicted data."""

    metrics = {}

    if len(pred_data) < 0.9 * num_eval_samples:
        logging.warning(r"Less than 90% of required eval samples supplied.")

    # Slice data to subset
    if num_eval_samples is None:
        num_eval_samples = min(len(pred_data), len(true_data))
    else:
        num_eval_samples = min(num_eval_samples, len(pred_data), len(true_data))
    true_data = true_data[:num_eval_samples]
    pred_data = pred_data[:num_eval_samples]
    metrics[f"{prefix}/num_eval_samples"] = min(num_eval_samples, len(pred_data))

    # Compute effective sample size
    if pred_data.logits is not None:
        ess = sampling_efficiency(pred_data.logits)
        metrics[f"{prefix}/effective_sample_size"] = ess

    if compute_distribution_distances:  # this is expensive - sometimes we don't want to compute it
        # Distribtuion distance metrics
        metrics.update(distribution_distances(true_data.samples, pred_data.samples, prefix=prefix))
        logging.info("Distance metrics computed")

    # Energy metrics
    metrics.update(
        energy_distances(
            true_data.energy,
            pred_data.energy,
            prefix=prefix,
        )
    )
    metrics[f"{prefix}/mean_energy"] = pred_data.energy.mean().cpu()
    logging.info("Energy metrics computed")

    # Ramachandran metrics
    metrics.update(ramachandran_metrics(true_data.samples, pred_data.samples, topology, prefix=prefix))
    logging.info("Ramachandran metrics computed")

    # TICA metric
    metrics.update(tica_metric(true_data.samples, pred_data.samples, topology, tica_model, prefix=prefix))
    logging.info("TICA metrics computed")

    # JSD metric
    metrics.update(jsd_metric(true_data.samples, pred_data.samples, topology, tica_model=tica_model, prefix=prefix))
    logging.info("JSD metrics computed")
    
    metrics.update(jsd_torus_metric(true_data.samples, pred_data.samples, topology, prefix=prefix))
    logging.info("JSD torus metrics computed")
    
    return metrics
