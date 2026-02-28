import torch

from src.evaluation.metrics.ess import normalized_ess
from src.evaluation.metrics.kmeans_jsd import tica_kmeans_jsd, torus_kmeans_jsd
from src.evaluation.metrics.wasserstein_distances import (
    energy_wasserstein,
    tica_wasserstein,
    torus_wasserstein,
)
from src.evaluation.plots.plot_atom_distances import plot_atom_distances
from src.evaluation.plots.plot_com_norms import plot_com_norms
from src.evaluation.plots.plot_energies import plot_energies
from src.evaluation.plots.plot_ramachandran import plot_ramachandran
from src.evaluation.plots.plot_tica import plot_tica
from src.utils.chirality import get_symmetry_change
from src.utils.dataclasses import EvalContext, SamplesData
from src.utils.pylogger import RankedLogger
from src.utils.standardization import destandardize_coords

logger = RankedLogger(__name__, rank_zero_only=False)


class PeptideEnsembleEvaluator:
    """Evaluates generated samples against reference data.

    Handles chirality fixing, metric computation, and plot generation.
    """

    NUM_EVAL_SAMPLES = 10_000

    def __init__(
        self,
        fix_symmetry: bool = True,
        drop_unfixable_symmetry: bool = False,
        do_plots: bool = True,
    ):
        self.fix_symmetry = fix_symmetry
        self.drop_unfixable_symmetry = drop_unfixable_symmetry
        self.do_plots = do_plots

    @torch.no_grad()  # eval-only: metric computation, no training
    def evaluate(
        self,
        samples_data_dict: dict[str, SamplesData],
        eval_context: EvalContext,
        log_image_fn=None,
        prefix: str = "",
    ) -> dict[str, float]:
        """Evaluate generated samples against reference data.

        Args:
            samples_data_dict: Dict mapping sample set names to SamplesData.
                Generated samples are in normalized space; unnormalized using
                eval_context.normalization_std.
            eval_context: EvalContext from datamodule.prepare_eval().
            log_image_fn: Optional image logging callable (img, title) -> None.
            prefix: String prefix for metric keys.

        Returns:
            Dict mapping metric names to computed values.
        """
        if len(prefix) > 0 and prefix[-1] != "/":
            prefix += "/"

        metrics = {}
        topology = eval_context.topology
        tica_model = eval_context.tica_model
        true_data = eval_context.true_data
        normalization_std = eval_context.normalization_std

        # Unnormalize generated sample sets for metrics/plots
        samples_data_dict = {
            name: SamplesData(
                destandardize_coords(data.samples, normalization_std),
                data.energy,
                logw=data.logw,
            )
            for name, data in samples_data_dict.items()
            if data is not None
        }

        # Fix chirality on proposal samples if present
        proposal_data = samples_data_dict.get("proposal")
        if proposal_data is not None:
            proposal_data, symmetry_metrics = self._fix_chirality(proposal_data, true_data, topology, prefix)
            metrics.update(symmetry_metrics)
            samples_data_dict = {**samples_data_dict, "proposal": proposal_data}

        # Plot true data
        if self.do_plots and log_image_fn is not None:
            plot_ramachandran(log_image_fn, true_data.samples, topology, prefix=prefix + "true")
            plot_tica(log_image_fn, true_data.samples, topology, tica_model, prefix=prefix + "true")

        # Compute metrics for each sample set
        for name, data in samples_data_dict.items():
            if data is None:
                continue

            if len(data) == 0:
                logger.warning(f"No {name} samples present.")
                continue

            logger.info(f"Evaluating {prefix + name} samples")

            # Slice to avoid computing metrics on too many samples
            data = data[: self.NUM_EVAL_SAMPLES * 2]

            metrics.update(
                self._evaluate_peptide_data(
                    true_data,
                    data,
                    topology=topology,
                    tica_model=tica_model,
                    prefix=prefix + name,
                )
            )

            if self.do_plots and log_image_fn is not None:
                plot_ramachandran(log_image_fn, data.samples, topology, prefix=prefix + name)
                plot_tica(log_image_fn, data.samples, topology, tica_model=tica_model, prefix=prefix + name)

        # Aggregate plots (energies, atom distances, CoM norms)
        if self.do_plots and log_image_fn is not None:
            # Reduce size so plotting doesn't crash with many samples
            true_plot = true_data[: self.NUM_EVAL_SAMPLES]
            plot_dict = {
                name: data[: self.NUM_EVAL_SAMPLES]
                for name, data in samples_data_dict.items()
                if data is not None and len(data) > 0
            }

            plot_energies(
                log_image_fn,
                true_plot.energy,
                {name: data.energy for name, data in plot_dict.items()},
                prefix=prefix,
            )
            plot_atom_distances(
                log_image_fn,
                true_plot.samples,
                {name: data.samples for name, data in plot_dict.items()},
                prefix=prefix,
            )
            plot_com_norms(
                log_image_fn,
                {name: data.samples for name, data in plot_dict.items()},
                prefix=prefix,
            )

        return metrics

    def _evaluate_peptide_data(
        self,
        true_data,
        pred_data,
        topology,
        tica_model,
        prefix: str = "",
    ):
        """Computes all metrics between true and predicted data."""
        metrics = {}
        num_eval_samples = self.NUM_EVAL_SAMPLES

        if len(pred_data) < 0.9 * num_eval_samples:
            logger.warning(r"Less than 90% of required eval samples supplied.")

        # Slice data to subset
        num_eval_samples = min(num_eval_samples, len(pred_data), len(true_data))
        true_data = true_data[:num_eval_samples]
        pred_data = pred_data[:num_eval_samples]
        metrics[f"{prefix}/num_eval_samples"] = min(num_eval_samples, len(pred_data))

        # Compute effective sample size
        if pred_data.logw is not None:
            ess = normalized_ess(pred_data.logw)
            metrics[f"{prefix}/effective_sample_size"] = ess

        metrics[f"{prefix}/mean_energy"] = pred_data.energy.mean().cpu()
        metrics[f"{prefix}/median_energy"] = pred_data.energy.median().cpu()

        metrics.update(energy_wasserstein(true_data.energy, pred_data.energy, prefix=prefix))
        logger.info("Energy wasserstein computed")

        metrics.update(torus_wasserstein(true_data.samples, pred_data.samples, topology, prefix=prefix))
        logger.info("Torus wasserstein computed")

        metrics.update(tica_wasserstein(true_data.samples, pred_data.samples, topology, tica_model, prefix=prefix))
        logger.info("TICA wasserstein computed")

        metrics.update(
            tica_kmeans_jsd(true_data.samples, pred_data.samples, topology, tica_model=tica_model, prefix=prefix)
        )
        metrics.update(torus_kmeans_jsd(true_data.samples, pred_data.samples, topology, prefix=prefix))
        logger.info("kMeans JSD computed")

        return metrics

    def _fix_chirality(
        self,
        pred_data: SamplesData,
        true_data: SamplesData,
        topology,
        prefix: str,
    ) -> tuple[SamplesData, dict]:
        """Detect and fix chirality issues in generated samples.

        Compares chirality signs between true and generated samples.
        If fix_symmetry is True, flips samples with incorrect chirality.
        If drop_unfixable_symmetry is True, removes samples that can't be fixed.

        Args:
            pred_data: Predicted samples (unnormalized).
            true_data: Reference samples (unnormalized).
            topology: mdtraj topology.
            prefix: Metric key prefix.

        Returns:
            Tuple of (fixed_proposal_data, symmetry_metrics).
        """
        metrics = {}

        samples = pred_data.samples.clone()

        first_symmetry_change = get_symmetry_change(
            true_data.samples,
            samples,
            topology,
        )

        correct_symmetry_rate = 1 - first_symmetry_change.float().mean().item()

        # Try flipping and check if it fixes the issue
        temp_samples = samples.clone()
        temp_samples[first_symmetry_change] *= -1

        second_symmetry_change = get_symmetry_change(
            true_data.samples,
            temp_samples,
            topology,
        )

        uncorrectable_symmetry_rate = second_symmetry_change.float().mean().item()

        metrics[f"{prefix}proposal/correct_symmetry_rate"] = correct_symmetry_rate
        metrics[f"{prefix}proposal/uncorrectable_symmetry_rate"] = uncorrectable_symmetry_rate

        if self.fix_symmetry:
            samples[first_symmetry_change] *= -1
            energy = pred_data.energy
            logw = pred_data.logw

            if self.drop_unfixable_symmetry:
                keep_mask = ~second_symmetry_change
                samples = samples[keep_mask]
                energy = energy[keep_mask]
                logw = logw[keep_mask] if logw is not None else None

            pred_data = SamplesData(samples, energy, logw=logw)

        return pred_data, metrics
