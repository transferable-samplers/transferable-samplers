from src.models.samplers.base_sampler import BaseSampler

class ProposalSampler(BaseSampler):
    def __init__(self):
        super().__init__()

    def sample(self, proposal_generator, source_energy, target_energy):

        self.start_timer()
        proposal_samples, proposal_log_q, prior_samples = proposal_generator(num_samples)
        metrics = self.timing_metrics(len(proposal_samples))
        samples_dict = {
            "samples": proposal_samples,
            "log_q_theta": proposal_log_q,
            "prior_samples": prior_samples,
        }
        self.save_samples_dict(samples_dict, "proposal")

        return ProposalSamples(proposal_samples, proposal_log_q), metrics

