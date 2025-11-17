import torch


def gaussian(x, n):
    shape = list(x.size())
    shape[0] *= n
    return torch.randn(shape, dtype=x.dtype, layout=x.layout, device=x.device)


def rademacher(x, n):
    shape = list(x.size())
    shape[0] *= n
    return torch.randint(low=0, high=2, size=shape, dtype=x.dtype, layout=x.layout, device=x.device).float() * 2 - 1.0


class TorchdynWrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format with additional dimension representing the change
    in likelihood over time."""

    def __init__(
        self,
        model,
        div_estimator="exact",
        logp_tol_scale=1.0,
        n_eps=1,
    ):
        super().__init__()
        self.model = model
        self.nfe = 0
        self.div_estimator = div_estimator
        self.logp_tol_scale = logp_tol_scale

        if div_estimator == "exact":
            self.div_fn = self.div_fn_exact
        elif div_estimator == "exact_no_functional":
            self.div_fn = self.div_fn_exact_no_functional
        elif div_estimator == "ito":
            self.div_fn = self.div_fn_exact
        else:
            self.div_fn = self.div_fn_hutch
            self.n = n_eps
            if div_estimator == "hutch_gaussian":
                self.eps_fn = gaussian
            elif div_estimator == "hutch_rademacher":
                self.eps_fn = rademacher
            else:
                raise NotImplementedError(f"likelihood estimator {div_estimator} is not implemented")

    def div_fn_hutch(self, t, x):
        """Hutchingson's trace estimator for the divergence of the vector field.

        Using Rademacher random variables for epsilons.
        """
        x = x.view(1, -1)  # batch dims required by EGNN architecture

        eps = self.eps_fn(x, self.n)

        def vecfield(y):
            return self.model(t, y)

        _, vjpfunc = torch.func.vjp(vecfield, x.repeat(self.n, 1))
        return (vjpfunc(eps)[0] * eps).sum() / self.n

    def div_fn_exact(self, t, x):
        def vecfield(y):
            y = y.view(1, -1)  # batch dims required by EGNN architecture
            return self.model(t, y).flatten()

        J = torch.func.jacrev(vecfield)

        return torch.trace(J(x))

    def div_fn_exact_no_functional(self, y, x):
        sum_diag = 0.0
        for i in range(y.shape[1]):
            sum_diag += torch.autograd.grad(y[:, i].sum(), x, create_graph=True)[0][:, i]
        return sum_diag

    def forward(self, t, x, *args, **kwargs):
        x = x[..., :-1]  # remove the divergence estimate

        if self.div_estimator == "exact_no_functional":
            with torch.enable_grad():
                x = x.requires_grad_(True)
                dx = self.model(t, x)
                dlog_p = -self.div_fn(dx, x)
        else:
            dx = self.model(t, x)
            dlog_p = -torch.vmap(self.div_fn, in_dims=(None, 0), randomness="different")(
                torch.tensor([t], device=x.device),
                x,
            )

        self.nfe += 1
        return torch.cat([dx, dlog_p[:, None] / self.logp_tol_scale], dim=-1).detach()


class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.nfe = 0

    def forward(self, t, x, *args, **kwargs):
        self.nfe += 1
        return self.model(t, x)
