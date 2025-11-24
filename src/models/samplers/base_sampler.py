class BaseSampler(ABC):
    def __init__(self, num_samples: int, output_dir: str, local_rank: int):
        self.num_samples = num_samples
        self.output_dir = output_dir
        self.local_rank = local_rank

    def save_samples_dict(self, samples_dict, prefix):
        if self.local_rank == 0:
            os.makedirs(f"{self.output_dir}/{prefix}", exist_ok=True)
            torch.save(samples_dict, f"{self.output_dir}/{prefix}/samples.pt")
            logging.info(f"Saving samples to {self.output_dir}/{prefix}/samples.pt")

    @abstractmethod
    def sample(self, proposal_generator, source_energy, target_energy):
        raise NotImplementedError
