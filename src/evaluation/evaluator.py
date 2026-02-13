import logging

import torch

from src.evaluation.metrics.evaluate_peptide_data import evaluate_peptide_data
from src.evaluation.plots.plot_atom_distances import plot_atom_distances
from src.evaluation.plots.plot_com_norms import plot_com_norms
from src.evaluation.plots.plot_energies import plot_energies
from src.evaluation.plots.plot_ramachandran import plot_ramachandran
from src.evaluation.plots.plot_tica import plot_tica
from src.utils.chirality import get_symmetry_change
from src.utils.dataclasses import EvalContext, SamplesData

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluates generated samples against reference data.

    Handles chirality fixing, metric computation, and plot generation.
    Extracted from TransferableBoltzmannGeneratorLitModule.evaluate()
    and BaseDataModule.metrics_and_plots().
    """

    def __init__(
        self,
        fix_symmetry: bool = True,
        drop_unfixable_symmetry: bool = False,
        num_eval_samples: int = 10_000,
        do_plots: bool = True,
    ):
        self.fix_symmetry = fix_symmetry
        self.drop_unfixable_symmetry = drop_unfixable_symmetry
        self.num_eval_samples = num_eval_samples
        self.do_plots = do_plots

    @torch.no_grad()
    def evaluate(
        self,
        sequence: str,
        samples_data_dict: dict[str, SamplesData],
        eval_context: EvalContext,
        log_image_fn=None,
        prefix: str = "",
    ) -> dict[str, float]:
        """Evaluate generated samples against reference data.

        Args:
            sequence: Peptide sequence string.
            samples_data_dict: Dict mapping sample set names to SamplesData,
                e.g. {"proposal": ..., "resampled": ..., "smc": ...}.
                Samples should be unnormalized.
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

        # Compute true data energy (eval_context.true_samples are normalized)
        true_data = SamplesData(
            eval_context.true_samples,  # already unnormalized in EvalContext? No.
            eval_context.target_energy_fn(eval_context.true_samples),
        )

        # Fix chirality on proposal samples if present
        proposal_data = samples_data_dict.get("proposal")
        if proposal_data is not None:
            proposal_data, symmetry_metrics = self._fix_chirality(
                proposal_data, true_data, topology, prefix
            )
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
            data = data[: self.num_eval_samples * 2]

            metrics.update(
                evaluate_peptide_data(
                    true_data,
                    data,
                    topology=topology,
                    tica_model=tica_model,
                    num_eval_samples=self.num_eval_samples,
                    prefix=prefix + name,
                    compute_distribution_distances=False,
                )
            )

            if self.do_plots and log_image_fn is not None:
                plot_ramachandran(log_image_fn, data.samples, topology, prefix=prefix + name)
                plot_tica(log_image_fn, data.samples, topology, tica_model=tica_model, prefix=prefix + name)

        # Aggregate plots (energies, atom distances, CoM norms)
        if self.do_plots and log_image_fn is not None:
            # Reduce size so plotting doesn't crash with many samples
            true_plot = true_data[: self.num_eval_samples]
            proposal_plot = samples_data_dict.get("proposal")
            resampled_plot = samples_data_dict.get("resampled")
            smc_plot = samples_data_dict.get("smc")

            proposal_plot = proposal_plot[: self.num_eval_samples] if proposal_plot is not None else None
            resampled_plot = resampled_plot[: self.num_eval_samples] if resampled_plot is not None else None
            smc_plot = smc_plot[: self.num_eval_samples] if smc_plot is not None else None

            plot_energies(
                log_image_fn,
                true_plot.energy,
                proposal_plot.energy if (proposal_plot is not None and len(proposal_plot) > 0) else None,
                resampled_plot.energy if (resampled_plot is not None and len(resampled_plot) > 0) else None,
                smc_plot.energy if (smc_plot is not None and len(smc_plot) > 0) else None,
                prefix=prefix,
            )
            plot_atom_distances(
                log_image_fn,
                true_plot.samples,
                proposal_plot.samples if (proposal_plot is not None and len(proposal_plot) > 0) else None,
                resampled_plot.samples if (resampled_plot is not None and len(resampled_plot) > 0) else None,
                smc_plot.samples if (smc_plot is not None and len(smc_plot) > 0) else None,
                prefix=prefix,
            )
            plot_com_norms(
                log_image_fn,
                proposal_plot.samples if (proposal_plot is not None and len(proposal_plot) > 0) else None,
                resampled_plot.samples if (resampled_plot is not None and len(resampled_plot) > 0) else None,
                smc_plot.samples if (smc_plot is not None and len(smc_plot) > 0) else None,
                prefix=prefix,
            )

        return metrics

    def _fix_chirality(
        self,
        proposal_data: SamplesData,
        true_data: SamplesData,
        topology,
        prefix: str,
    ) -> tuple[SamplesData, dict]:
        """Detect and fix chirality issues in generated samples.

        Compares chirality signs between true and generated samples.
        If fix_symmetry is True, flips samples with incorrect chirality.
        If drop_unfixable_symmetry is True, removes samples that can't be fixed.

        Args:
            proposal_data: Generated samples (unnormalized).
            true_data: Reference samples (unnormalized).
            topology: mdtraj topology.
            prefix: Metric key prefix.

        Returns:
            Tuple of (fixed_proposal_data, symmetry_metrics).
        """
        metrics = {}

        samples = proposal_data.samples.clone()

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
            energy = proposal_data.energy
            logits = proposal_data.logits

            if self.drop_unfixable_symmetry:
                keep_mask = ~second_symmetry_change
                samples = samples[keep_mask]
                energy = energy[keep_mask]
                logits = logits[keep_mask] if logits is not None else None

            proposal_data = SamplesData(samples, energy, logits=logits)

        return proposal_data, metrics
