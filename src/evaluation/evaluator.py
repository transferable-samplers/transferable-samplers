        def add_aggregate_metrics(self, metrics: dict[str, torch.Tensor], prefix: str = "val") -> dict[str, torch.Tensor]:
        """Aggregate metrics across all sequences."""

        mean_dict_list = defaultdict(list)
        median_dict_list = defaultdict(list)
        count_dict = defaultdict(int)

        # Parse and aggregate metrics along peptide sequences
        for key, value in metrics.items():
            if key.startswith(prefix):  # TODO not sure this is needed here
                # Extract sequence and metric name
                parts = key.split("/")
                metric_name = "/".join(parts[2:])

                # Add to mean and median dictionaries
                mean_key = f"{prefix}/mean/{metric_name}"
                median_key = f"{prefix}/median/{metric_name}"
                count_key = f"{prefix}/count/{metric_name}"

                if isinstance(value, torch.Tensor):
                    value = value.item()
                elif isinstance(value, (int, float)):
                    value = float(value)

                mean_dict_list[mean_key].append(value)
                median_dict_list[median_key].append(value)
                count_dict[count_key] += 1

        # Compute mean and median for each metric
        mean_dict = {}
        median_dict = {}
        for key, value in mean_dict_list.items():
            mean_dict[key] = stats.mean(value)

        for key, value in median_dict_list.items():
            median_dict[key] = stats.median(value)

        metrics.update(mean_dict)
        metrics.update(median_dict)
        metrics.update(count_dict)
        return metrics

    def detach_and_cpu(
        self, obj
    ):  # TODO hack to have this here? at all? you could just be more careful to detach / cpu?
        """
        Recursively detach and move all tensors to CPU within a nested structure.
        Works with dicts, lists, tuples, and tensors.
        """
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu()
        elif isinstance(obj, dict):
            return {k: self.detach_and_cpu(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.detach_and_cpu(v) for v in obj]
        elif isinstance(obj, tuple):
            return tuple(self.detach_and_cpu(v) for v in obj)
        else:
            return obj  # Leave other data types (int, float, str, etc.) as-is

    def save_samples_dict(self, samples_dict, prefix):
        if self.local_rank == 0:
            os.makedirs(f"{self.output_dir}/{prefix}", exist_ok=True)
            torch.save(samples_dict, f"{self.output_dir}/{prefix}/samples.pt")
            logging.info(f"Saving samples to {self.output_dir}/{prefix}/samples.pt")


class Evaluator:

    @torch.no_grad()
    def evaluate_sequence(
        self,
        sequence,
        evaluation_inputs,
        model_inputs,
        energy_fn,
        prefix: str = "val",
        proposal_generator=None,
        output_dir=None,
    ) -> None:
        """Generates samples from the proposal and runs SMC if enabled.
        Also computes metrics and plots using the standalone metrics_and_plots function.
        """

        metrics = {}

        true_data = SamplesData(
            self.datamodule.unnormalize(evaluation_inputs.true_samples),
            energy_fn(evaluation_inputs.true_samples),
        )

        # Compute proposal center of mass std
        coms = proposal_data.samples.mean(dim=1, keepdim=False)
        proposal_com_std = coms.std()

        if self.use_com_adjustment:
            proposal_log_q = proposal_data.log_q_theta + com_energy_adjustment(proposal_data.samples, proposal_com_std)

        proposal_samples_energy = target_energy(proposal_data.samples)


        # Define proposal generator
        if proposal_generator is None:
            proposal_generator = self.batched_generate_samples
            if "dummy_ll" in self.hparams and self.hparams.dummy_ll:
                proposal_generator = lambda x: self.batched_generate_samples(x, dummy_ll=True)

        if prefix.startswith("test"):
            num_proposal_samples = self.hparams.sampling_config.num_test_proposal_samples
        else:
            num_proposal_samples = self.hparams.sampling_config.num_proposal_samples

        proposal_samples_energy = energy_fn(proposal_samples)

        # Datatype for easier metrics and plotting
        proposal_data = SamplesData(
            self.datamodule.unnormalize(proposal_samples),
            proposal_samples_energy,
        )

        # Compute proposal center of mass std
        coms = proposal_samples.mean(dim=1, keepdim=False)
        proposal_com_std = coms.std()
        # TODO little scary relying on this class attribute! - gets used in self.proposal_energy
        # when use_com_adjustment=True
        self.proposal_com_std = proposal_com_std
        logging.info(f"Proposal CoM std: {proposal_com_std}")
        self.log(f"{prefix}/proposal_com_std", proposal_com_std, sync_dist=True)

        # Datatype for easier metrics and plotting
        proposal_data = SamplesData(
            self.datamodule.unnormalize(proposal_samples),
            proposal_samples_energy,
        )


        reweighted_data = SamplesData(
            self.datamodule.unnormalize(proposal_samples[resampling_index]),
            proposal_samples_energy[resampling_index],
            logits=resampling_logits,
        )

        if self.hparams.sampling_config.get("load_samples_path", None) is not None:
            load_samples_path_smc = self.hparams.sampling_config.load_samples_path.replace("samples", "smc_samples")
        else:
            load_samples_path_smc = None

        if load_samples_path_smc and os.path.exists(load_samples_path_smc):
            logging.info(f"Loading SMC samples from {load_samples_path_smc}")
            smc_samples_dict = torch.load(load_samples_path_smc, map_location=self.device)
            smc_samples = smc_samples_dict["samples"]
            smc_logits = smc_samples_dict["logits"]
            smc_data = SamplesData(
                self.datamodule.unnormalize(smc_samples),
                energy_fn(smc_samples),
                logits=smc_logits,
            )
        elif self.smc_sampler is not None and self.smc_sampler.enabled:
            logging.info("SMC sampling enabled")

            num_smc_samples = min(self.hparams.sampling_config.num_smc_samples, len(proposal_samples))

            # Generate smc samples and record time
            torch.cuda.synchronize()
            start_time = time.time()

            # TODO: Make conditional proposal energy
            cond_proposal_energy = lambda _x: self.proposal_energy(_x, permutations=permutations, encodings=encodings)
            smc_samples, smc_logits = self.smc_sampler.sample(
                proposal_samples[:num_smc_samples], cond_proposal_energy, energy_fn
            )  # already returned resampled
            torch.cuda.synchronize()
            time_duration = time.time() - start_time
            self.log(f"{prefix}/smc/samples_walltime", time_duration, sync_dist=True)
            self.log(f"{prefix}/smc/samples_per_second", len(smc_samples) / time_duration, sync_dist=True)

            # Save samples to disk
            smc_samples_dict = {
                "smc_samples": smc_samples,
                "smc_logits": smc_logits,
            }
            if self.local_rank == 0:
                os.makedirs(f"{self.output_dir}/{prefix}", exist_ok=True)

                torch.save(smc_samples_dict, f"{self.output_dir}/{prefix}/smc_samples.pt")
                logging.info(f"Saving {len(smc_samples)} samples to {self.output_dir}/{prefix}_smc_samples.pt")

            # Datatype for easier metrics and plotting
            smc_data = SamplesData(
                self.datamodule.unnormalize(smc_samples),
                energy_fn(smc_samples),
                logits=smc_logits,
            )
        else:
            smc_data = None

        if self.local_rank == 0:
            # log dataset metrics
            metrics.update(
                metrics_and_plots(
                    log_image_fn=self.log_image,
                    sequence=sequence,
                    topology=evaluation_inputs.topology,
                    tica_model=evaluation_inputs.tica_model,
                    num_eval_samples=evaluation_inputs.num_eval_samples,
                    true_data=true_data,
                    proposal_data=proposal_data,
                    resampled_data=reweighted_data,
                    smc_data=smc_data,
                    do_plots=evaluation_inputs.do_plots,
                    prefix=prefix,
                )
            )
        else:
            metrics = {}
        return metrics

        def evaluate_all(self, prefix):
        metrics = {}
        eval_sequences = self.datamodule.val_sequences if prefix.startswith("val") else self.datamodule.test_sequences
        for sequence in eval_sequences:
            # TODO: single peptides expects prefix as input while transferable expects sequence as input
            model_inputs, evaluation_inputs, energy_fn = self.datamodule.prepare_eval(
                prefix=prefix, sequence=sequence
            )
            logging.info(f"Evaluating {sequence} samples")
            metrics.update(
                self.evaluate(
                    sequence,
                    evaluation_inputs,
                    model_inputs,
                    energy_fn,
                    prefix=f"{prefix}/{sequence}",
                    proposal_generator=self.batched_generate_samples,
                )
            )

        # Aggregate metrics across all sequences
        if self.local_rank == 0:
            metrics = self.detach_and_cpu(metrics)  # Ensure all tensors are detached and on CPU
            metric_object_list = [self.add_aggregate_metrics(metrics, prefix=prefix)]
        else:
            metric_object_list = [None]  # List must have same length for broadcast
        if self.trainer.world_size > 1:
            # Broadcast metrics to all processes - must log from all for checkpointing
            torch.distributed.broadcast_object_list(metric_object_list, src=0)
        self.log_dict(metric_object_list[0])