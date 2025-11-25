import logging
from typing import Callable, Optional

from src.evaluation.chirality import fix_chirality
from src.evaluation.metrics.distribution_distances import energy_distances
from src.evaluation.metrics.ess import sampling_efficiency
from src.evaluation.metrics.jsd_metric import jsd_metric
from src.evaluation.metrics.ramachandran import ramachandran_metrics
from src.evaluation.metrics.tica_metric import tica_metric
from src.evaluation.plots.plot_atom_distances import plot_atom_distances
from src.evaluation.plots.plot_com_norms import plot_com_norms
from src.evaluation.plots.plot_energies import plot_energies
from src.evaluation.plots.plot_ramachandran import plot_ramachandran
from src.evaluation.plots.plot_tica import plot_tica
from src.utils.dataclasses import SamplesData

logger = logging.getLogger(__name__)


class Evaluator:
    def __init__(self, fix_symmetry: bool = True, drop_unfixable_symmetry: bool = False, num_eval_samples: int = 10_000):
        self.fix_symmetry = fix_symmetry
        self.drop_unfixable_symmetry = drop_unfixable_symmetry
        self.num_eval_samples = num_eval_samples

    def compute_metrics(self, true_data: SamplesData, pred_data: SamplesData, topology, tica_model, num_eval_samples: int, prefix: str = ""):
        """Computes all metrics between true and predicted data."""

        metrics = {}

        if len(pred_data) < 0.9 * num_eval_samples:
            logger.warning("Less than 90% of required eval samples supplied.")

        # Slice data to subset
        if num_eval_samples is None:
            actual_num_eval_samples = min(len(pred_data), len(true_data))
        else:
            actual_num_eval_samples = min(num_eval_samples, len(pred_data), len(true_data))
        true_data_sliced = SamplesData(
            x=true_data.x[:actual_num_eval_samples],
            proposal_energy=true_data.proposal_energy[:actual_num_eval_samples] if true_data.proposal_energy is not None else None,
            target_energy=true_data.target_energy[:actual_num_eval_samples],
            importance_logits=true_data.importance_logits[:actual_num_eval_samples] if true_data.importance_logits is not None else None,
        )
        pred_data_sliced = SamplesData(
            x=pred_data.x[:actual_num_eval_samples],
            proposal_energy=pred_data.proposal_energy[:actual_num_eval_samples] if pred_data.proposal_energy is not None else None,
            target_energy=pred_data.target_energy[:actual_num_eval_samples],
            importance_logits=pred_data.importance_logits[:actual_num_eval_samples] if pred_data.importance_logits is not None else None,
        )
        metrics[f"{prefix}/num_eval_samples"] = min(actual_num_eval_samples, len(pred_data_sliced))

        # Compute effective sample size
        if pred_data_sliced.importance_logits is not None:
            ess = sampling_efficiency(pred_data_sliced.importance_logits)
            metrics[f"{prefix}/effective_sample_size"] = ess

        # Energy metrics
        metrics.update(
            energy_distances(
                true_data_sliced.target_energy,
                pred_data_sliced.target_energy,
                prefix=prefix,
            )
        )
        metrics[f"{prefix}/mean_energy"] = pred_data_sliced.target_energy.mean().cpu()
        metrics[f"{prefix}/median_energy"] = pred_data_sliced.target_energy.median().cpu()
        logger.info("Energy metrics computed")

        # Ramachandran metrics
        metrics.update(ramachandran_metrics(true_data_sliced.x, pred_data_sliced.x, topology, prefix=prefix))
        logger.info("Ramachandran metrics computed")

        # TICA metric
        metrics.update(tica_metric(true_data_sliced.x, pred_data_sliced.x, topology, tica_model, prefix=prefix))
        logger.info("TICA metrics computed")

        # JSD metric
        metrics.update(jsd_metric(true_data_sliced.x, pred_data_sliced.x, topology, tica_model=tica_model, prefix=prefix))
        logger.info("JSD metrics computed")

        return metrics


    def evaluate(self, sequence, samples_data_dict, evaluation_inputs, target_energy_fn):
        """
        Compute evaluation metrics and log diagnostic plots for a single peptide sequence.

        Logs Ramachandran plots, TICA projections, energy distributions, atom
        distance distributions, and center-of-mass norms for provided datasets.
        Also computes quantitative evaluation metrics by comparing generated
        samples against true reference data.

        Args:
            sequence (str): Peptide sequence identifier.
            samples_data_dict (dict): Dictionary mapping sample type names to SamplesData objects.
            evaluation_inputs: EvaluationInputs dataclass containing true_samples, topology, tica_model, etc.
            target_energy_fn (Callable): Function to compute energy for samples.

        Returns:
            tuple: (metrics_dict, plots_dict) containing computed metrics and plots.
        """

        topology = evaluation_inputs.topology
        tica_model = evaluation_inputs.tica_model
        true_samples = evaluation_inputs.true_samples
        num_eval_samples = evaluation_inputs.num_eval_samples
        do_plots = evaluation_inputs.do_plots

        # Create true SamplesData for comparison
        true_data = SamplesData(
            x=true_samples,
            proposal_energy=None,
            target_energy=target_energy_fn(true_samples),
            importance_logits=None,
        )

        metrics = {}
        plots = {}

        if self.fix_symmetry:
            for name, samples_data in samples_data_dict.items():
                samples_data_dict[name] = fix_chirality(
                    true_samples, 
                    samples_data, 
                    topology,
                    drop_unfixable=self.drop_unfixable_symmetry
                )

        for name, samples_data in samples_data_dict.items():
            logger.info(f"Evaluating {sequence}/{name} samples")
            if len(samples_data) == 0:
                logger.warning(f"No {name} samples present.")
                continue

            sequence_metrics = self.compute_metrics(
                true_data,
                samples_data,
                topology,
                tica_model,
                num_eval_samples=num_eval_samples,
                prefix=f"{sequence}/{name}",
            )
            metrics.update(sequence_metrics)
            
            if do_plots:
                plots[f"{sequence}/{name}"] = {
                    "ramachandran": plot_ramachandran(samples_data.x, topology, prefix=f"{sequence}/{name}"),
                    "tica": plot_tica(
                        samples_data.x,
                        topology,
                        tica_model=tica_model,
                        prefix=f"{sequence}/{name}",
                    ),
                }

        if do_plots:
            plots[f"{sequence}/energies"] = plot_energies(
                samples_data_dict,
                num_samples=num_eval_samples,
                prefix=f"{sequence}",
            )
            plots[f"{sequence}/atom_distances"] = plot_atom_distances(
                samples_data_dict,
                num_samples=num_eval_samples,
                prefix=f"{sequence}",
            )
            plots[f"{sequence}/com_norms"] = plot_com_norms(
                samples_data_dict,
                num_samples=num_eval_samples,
                prefix=f"{sequence}",
            )

        return metrics, plots


