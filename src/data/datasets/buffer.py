import torch


def sample_without_replacement(logits: torch.Tensor, n: int) -> torch.Tensor:
    # https://timvieira.github.io/blog/post/2014/07/31/gumbel-max-trick/
    z = torch.distributions.Gumbel(torch.tensor(0.0), torch.tensor(1.0)).sample(logits.shape).to(logits.device)
    topk = torch.topk(z + logits, n, sorted=False)
    indices = topk.indices
    indices = indices[torch.randperm(n).to(indices.device)]
    return indices


class ReplayBuffer:
    def __init__(
        self,
        max_length: int,
        sample_with_replacement: bool = False,
    ):
        """
        Create prioritised replay buffer for batched sampling and adding of data.
        Args:
            dim: dimension of x data
            max_length: maximum length of the buffer
            min_sample_length: minimum length of buffer required for sampling
            initial_sampler: sampler producing x, log_w and log q, used to fill the buffer up to
                the min sample length. The initialised flow + AIS may be used here,
                or we may desire to use AIS with more distributions to give the flow a "good start".
            device: replay buffer device
            sample_with_replacement: Whether to sample from the buffer with replacement.
            fill_buffer_during_init: Whether to use `initial_sampler` to fill the buffer initially.
                If a checkpoint is going to be loaded then this should be set to False.

        The `max_length` and `min_sample_length` should be sufficiently long to prevent overfitting
        to the replay data. For example, if `min_sample_length` is equal to the
        sampling batch size, then we may overfit to the first batch of data, as we would update
        on it many times during the start of training.
        """
        self.max_length = max_length
        self.buffer = []
        self.seq_name = None
        self.sample_with_replacement = sample_with_replacement

    def __len__(self):
        return len(self.buffer)

    @torch.no_grad()
    def add(self, x: torch.Tensor, seq_name: str = None) -> None:
        """Add a new batch of generated data to the replay buffer."""
        x = x.detach().cpu()
        if len(self.buffer) == 0:
            self.buffer = x
        else:
            self.buffer = torch.concat([self.buffer, x], dim=0)

        self.seq_name = seq_name
        self.buffer = self.buffer[-self.max_length :]

    @torch.no_grad()
    def sample(
        self,
        idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a batch of sampled data, if the batch size is specified then the batch will have
        a leading axis of length batch_size, otherwise the default self.batch_size will be used."""

        local_idx = idx % len(self.buffer)

        x, seq_name = (
            self.buffer[local_idx],
            self.seq_name,
        )
        return x, seq_name

    def save(self, path):
        """Save buffer to file."""
        to_save = {
            "x": self.buffer.detach().cpu(),
            "seq_name": self.seq_name,
        }
        torch.save(to_save, path)

    def load(self, path):
        """Load buffer from file."""
        old_buffer = torch.load(path)
        self.buffer = old_buffer["x"]
        self.seq_name = old_buffer["seq_name"]
