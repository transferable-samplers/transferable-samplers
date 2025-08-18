import torch
from einops import rearrange
from torch import einsum


def exists(val) -> bool:
    """returns whether val is not none"""
    return val is not None


def default(x, y):
    """returns x if it exists, otherwise y"""
    return x if exists(x) else y


max_neg_value = lambda x: torch.finfo(x.dtype).min


class Attention(torch.nn.Module):
    USE_SPDA = True

    def __init__(
        self,
        in_channels: int,
        head_channels: int,
        use_attn_pair_bias: bool = False,  # false for consistency with tarflow.py
        use_qkln: bool = False,  # false for consistency with tarflow.py
        dropout: float = 0.0,
    ):
        assert in_channels % head_channels == 0
        super().__init__()
        self.norm = torch.nn.LayerNorm(in_channels)
        self.qkv = torch.nn.Linear(in_channels, in_channels * 3)
        self.proj = torch.nn.Linear(in_channels, in_channels)
        self.num_heads = in_channels // head_channels
        self.sqrt_scale = head_channels ** (-0.25)
        self.sample = False

        self.use_attn_pair_bias = use_attn_pair_bias

        self.q_layer_norm = torch.nn.LayerNorm(in_channels) if use_qkln else torch.nn.Identity()
        self.k_layer_norm = torch.nn.LayerNorm(in_channels) if use_qkln else torch.nn.Identity()

        self.dropout = dropout

        self.k_cache: dict[str, list[torch.Tensor]] = {"cond": [], "uncond": []}
        self.v_cache: dict[str, list[torch.Tensor]] = {"cond": [], "uncond": []}

    def forward_spda(
        self,
        x,
        pair: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        temp: float = 1.0,
        which_cache: str = "cond",
    ):
        if (self.use_attn_pair_bias and not exists(pair)) or (not self.use_attn_pair_bias and exists(pair)):
            raise ValueError("pair must be provided if use_attn_pair_bias is True")

        B, T, C = x.size()
        x = self.norm(x.float()).type(x.dtype)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = self.q_layer_norm(q)
        k = self.k_layer_norm(k)

        # if provided, project pair from (B, seq_len, seq_len, C) to (B, seq_len, seq_len, num_heads)
        # and then rearrange to (B, num_heads, seq_len, seq_len).
        # if not provided, set bias to 0.0
        bias = rearrange(pair, "b ... h -> b h ...") if (exists(pair) and self.use_attn_pair_bias) else 0.0
        q, k, v = map(lambda t: rearrange(t, "b ... (h d) -> b h ... d", h=self.num_heads), (q, k, v))

        if self.sample:
            self.k_cache[which_cache].append(k)
            self.v_cache[which_cache].append(v)
            k = torch.cat(self.k_cache[which_cache], dim=2)  # note that sequence dimension is now 2
            v = torch.cat(self.v_cache[which_cache], dim=2)

        scale = self.sqrt_scale**2 / temp
        if mask is not None:
            mask = mask.bool()

        if self.use_attn_pair_bias:
            if mask is not None:
                if mask.ndim < 4:
                    mask = mask.reshape(*([1] * (4 - mask.ndim)), *mask.shape)

                attn_mask = torch.zeros_like(mask, dtype=q.dtype)
                attn_mask.masked_fill_(mask.logical_not(), float("-inf"))
                mask = attn_mask + bias
            else:
                mask = bias

        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=scale, dropout_p=self.dropout if self.training else 0.0
        )
        x = x.transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        return x

    def forward_base(
        self,
        x,
        pair: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        temp: float = 1.0,
        which_cache: str = "cond",
    ):
        if (self.use_attn_pair_bias and not exists(pair)) or (not self.use_attn_pair_bias and exists(pair)):
            print(self.use_attn_pair_bias, exists(pair))
            raise ValueError("pair must be provided if use_attn_pair_bias is True")

        x = self.norm(x.float()).type(x.dtype)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = self.q_layer_norm(q)
        k = self.k_layer_norm(k)

        bias = rearrange(pair, "b ... h -> b h ...") if (exists(pair) and self.use_attn_pair_bias) else 0.0

        q, k, v = map(lambda t: rearrange(t, "b ... (h d) -> b h ... d", h=self.num_heads), (q, k, v))
        if self.sample:
            self.k_cache[which_cache].append(k)
            self.v_cache[which_cache].append(v)
            k = torch.cat(self.k_cache[which_cache], dim=2)
            v = torch.cat(self.v_cache[which_cache], dim=2)

        x = self._attn(q, k, v, bias, mask, temp)

        # I don't know why there is a sigmoid here in original proteina.
        # It does not show in their figure nor talked about in the paper.
        # This causes the output of the forward to NOT match the outputs of the other
        # forward attention functions (base and spda)
        # x = torch.sigmoid(g) * x
        x = rearrange(x, "b h n d -> b n (h d)", h=self.num_heads)
        x = self.proj(x)
        return x

    def _attn(self, q, k, v, bias, mask: torch.Tensor | None = None, temp: float = 1.0):
        """Perform attention update"""
        scale = self.sqrt_scale**2 / temp
        sim = einsum("b h i d, b h j d -> b h i j", q, k) * scale
        sim += bias
        if exists(mask):
            mask = mask.bool()
            attn_mask = torch.zeros_like(mask, dtype=q.dtype)
            attn_mask.masked_fill_(mask.logical_not(), max_neg_value(sim))
            sim = (sim + attn_mask).float()

        attn = torch.softmax(sim, dim=-1).type(sim.dtype)
        attn = torch.dropout(attn, self.dropout, train=self.sample)
        return einsum("b h i j, b h j d -> b h i d", attn, v)

    def forward(
        self,
        x,
        pair: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        temp: float = 1.0,
        which_cache: str = "cond",
    ):
        if self.USE_SPDA:
            return self.forward_spda(x, pair=pair, mask=mask, temp=temp, which_cache=which_cache)

        return self.forward_base(x, pair=pair, mask=mask, temp=temp, which_cache=which_cache)


class MLP(torch.nn.Module):
    def __init__(self, channels: int, expansion: int):
        super().__init__()
        self.norm = torch.nn.LayerNorm(channels)
        self.main = torch.nn.Sequential(
            torch.nn.Linear(channels, channels * expansion),
            torch.nn.GELU(),
            torch.nn.Linear(channels * expansion, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(self.norm(x.float()).type(x.dtype))


class AttentionBlock(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        head_channels: int,
        expansion: int = 4,
        use_qkln: bool = False,
        use_attn_pair_bias: bool = True,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.attention = Attention(
            in_channels=channels,
            head_channels=head_channels,
            use_attn_pair_bias=use_attn_pair_bias,
            use_qkln=use_qkln,
            dropout=dropout,
        )

        self.mlp = MLP(channels, expansion)

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor | None = None,
        pair: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        attn_temp: float = 1.0,
        which_cache: str = "cond",
    ) -> torch.Tensor:
        if mask is None:
            mask = torch.ones(x.shape[:2], device=x.device, dtype=torch.bool)

        if cond is not None:
            x = x + cond

        x = x * mask[..., None]
        x = x + self.attention(x, pair=pair, mask=attn_mask, temp=attn_temp, which_cache=which_cache)
        x = x + self.mlp(x)
        return x * mask[..., None]


if __name__ == "__main__":
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_printoptions(sci_mode=True, precision=2)
    torch.manual_seed(1)

    attn = Attention(128, 64, use_qkln=True, use_attn_pair_bias=True)

    x = torch.randn((128, 10, 128))
    mask = torch.tril(torch.ones((x.shape[1], x.shape[1]), dtype=torch.bool))
    pair = torch.randn((128, 10, 10, 128))
    y = attn(x, pair=pair, mask=mask)

    attn.USE_SPDA = False
    z = attn(x, pair=pair, mask=mask)

    error = (abs(y - z)).mean()
    print(f"Error between SPDA and proteina: {error}")
    assert torch.allclose(y, z, atol=1e-6), f"Error: {error}"

    w = attn(x, pair=pair, mask=mask)
    assert not torch.isnan(w).any()
    assert not torch.isinf(w).any()

    # pair = torch.randn((128, 10, 10, 128))
    # w = attn(x, pair=pair, mask=mask)
    # assert not torch.allclose(w, z)
