def get_logit_clip_mask(logits: torch.Tensor, clip_threshold: float) -> torch.Tensor:
    logit_clip_mask = logits < torch.quantile(
        logits,
        1 - float(self.logit_clip_filter_pct) / 100.0,
    )
    return logit_clip_mask

def compute_com_std(x: torch.Tensor) -> float:
    com = x.mean(dim=1, keepdim=False)
    com_std = com.std(dim=-1)
    return com_std

def com_energy_adjustment(x: torch.Tensor, sigma: float) -> torch.Tensor:
    com = x.mean(dim=1, keepdim=False)
    com_norm = com.norm(dim=-1)
    com_energy = com_norm**2 / (2 * sigma**2) - torch.log(
        com_norm**2 / (math.sqrt(2) * sigma**3 * scipy.special.gamma(3 / 2))
    )
    return com_energy