class BaseSampler(ABC):
    def __init__(self, num_samples: int, logit_clip_filter_pct: float = 0.0):
        self.num_samples = num_samples
        self.logit_clip_filter_pct = logit_clip_filter # Applied to the initial proposal samples

    def get_logit_clip_mask(self, logits: torch.Tensor, clip_threshold: float) -> torch.Tensor:
        logit_clip_mask = logits < torch.quantile(
            logits,
            1 - float(self.logit_clip_filter_pct) / 100.0,
        )
        return logit_clip_mask

    def get_resampling_index(logits):
        probs = torch.softmax(logits, dim=-1)
        resampling_index = torch.multinomial(probs, logits.size(0), replacement=True)
        return resampling_index

    @abstractmethod
    def sample(self, proposal_generator, source_energy, target_energy):
        raise NotImplementedError
