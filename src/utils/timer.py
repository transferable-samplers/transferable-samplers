
import time
import torch

class Timer:
    def __init__(self):
        self.start_time = None
        self.elapsed_time = None

    def start_timer(self):
        torch.cuda.synchronize()
        self.start_time = time.time()

    def stop_timer(self):
        torch.cuda.synchronize()
        self.elapsed_time = time.time() - self.start_time

    def timing_metrics(self, num_samples):
        return {
            f"samples_walltime": self.elapsed_time,
            f"samples_per_second": num_samples / self.elapsed_time,
            f"seconds_per_sample": self.elapsed_time / num_samples,
        }