from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp

DELTA_MIN = 1e-3
DELTA_MAX = 1e-1
DELTA_FLOOR = 1e-4


def _delta_bias_initializer(
    min_dt: float = DELTA_MIN,
    max_dt: float = DELTA_MAX,
    floor: float = DELTA_FLOOR,
):
    def init(key: jax.Array, shape: tuple[int, ...], dtype=jnp.float32) -> jax.Array:
        if len(shape) != 1:
            raise RuntimeError(f"Delta bias must be rank-1, got shape={shape}")
        log_dt = jax.random.uniform(
            key,
            shape,
            minval=jnp.log(jnp.asarray(min_dt, dtype=dtype)),
            maxval=jnp.log(jnp.asarray(max_dt, dtype=dtype)),
            dtype=dtype,
        )
        dt = jnp.maximum(jnp.exp(log_dt), jnp.asarray(floor, dtype=dtype))
        return dt + jnp.log(-jnp.expm1(-dt))

    return init


def _pad_seq_dim(x: jax.Array, pad_size: int) -> jax.Array:
    if pad_size == 0:
        return x
    pad_width = [(0, 0)] * x.ndim
    pad_width[1] = (0, pad_size)
    return jnp.pad(x, pad_width, mode="constant", constant_values=0.0)


def _segsum(x: jax.Array) -> jax.Array:
    t = x.shape[-1]
    x_cumsum = jnp.cumsum(x, axis=-1)
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
    mask = jnp.tril(jnp.ones((t, t), dtype=bool), k=0)
    return jnp.where(mask, x_segsum, -jnp.inf)


def _ssd_forward(
    *,
    x: jax.Array,
    dt: jax.Array,
    a: jax.Array,
    b_mat: jax.Array,
    c_mat: jax.Array,
    chunk_size: int,
    d_skip: jax.Array,
    dt_bias: jax.Array,
    dt_min: float,
    dt_max: float,
    initial_states: jax.Array | None,
) -> tuple[jax.Array, jax.Array]:
    """Chunked SSD scan.

    Mamba-2 uses a single SSM group per layer, i.e. b_mat/c_mat are shared
    across all heads. They are kept at shape (batch, seq_len, state_size)
    throughout this function -- the head dimension is never materialized for
    them. All broadcasting over heads happens implicitly inside the
    `jnp.einsum` calls below (the head axis 'h' is simply omitted from
    b_mat's/c_mat's einsum subscript), which avoids ever allocating a
    (batch, seq_len, num_heads, state_size)-shaped array.
    """
    _, seq_len, num_heads, _ = x.shape
    pad_size = (chunk_size - seq_len % chunk_size) % chunk_size

    dt = jax.nn.softplus(dt + dt_bias[None, None, :])
    # softplus's output already lies in (0, inf), so with the default bounds
    # (dt_min=0.0, dt_max=inf) this clip is a mathematical no-op. XLA can't
    # infer that on its own, so skip the op entirely in that (default) case
    # rather than paying for a full elementwise pass that changes nothing.
    # dt_min/dt_max are plain Python floats here (module attributes, not
    # traced arrays), so this branch is resolved at trace time.
    if dt_min > 0.0 or dt_max < float("inf"):
        dt = jnp.clip(dt, dt_min, dt_max)

    x_padded = _pad_seq_dim(x, pad_size)
    dt_padded = _pad_seq_dim(dt, pad_size)
    b_padded = _pad_seq_dim(b_mat, pad_size)
    c_padded = _pad_seq_dim(c_mat, pad_size)

    padded_len = x_padded.shape[1]
    num_chunks = padded_len // chunk_size

    alpha = jnp.exp(a.astype(dt_padded.dtype)[None, None, :] * dt_padded)
    alpha_chunks = alpha.reshape(x_padded.shape[0], num_chunks, chunk_size, num_heads)
    dt_chunks = dt_padded.reshape(x_padded.shape[0], num_chunks, chunk_size, num_heads)
    b_chunks = b_padded.reshape(
        x_padded.shape[0], num_chunks, chunk_size, b_padded.shape[-1]
    )
    c_chunks = c_padded.reshape(
        x_padded.shape[0], num_chunks, chunk_size, c_padded.shape[-1]
    )
    x_chunks = x_padded.reshape(
        x_padded.shape[0], num_chunks, chunk_size, num_heads, x_padded.shape[3]
    )

    # Transpose straight to (num_chunks, chunk_size, batch, ...) in one shot.
    # The scan body below previously re-swapped chunk_size<->batch on every
    # one of the num_chunks iterations to reach this same layout -- composing
    # two transposes (this one, plus that per-iteration swap) to land on a
    # layout that a single permutation already gets to. Same total elements,
    # roughly half the transpose traffic, and six fewer transpose ops living
    # inside the scanned hot loop.
    alpha_scan = jnp.transpose(alpha_chunks, (1, 2, 0, 3))
    dt_scan = jnp.transpose(dt_chunks, (1, 2, 0, 3))
    b_scan = jnp.transpose(b_chunks, (1, 2, 0, 3))
    c_scan = jnp.transpose(c_chunks, (1, 2, 0, 3))
    x_scan = jnp.transpose(x_chunks, (1, 2, 0, 3, 4))

    if initial_states is None:
        init_state = jnp.zeros(
            (
                x_padded.shape[0],
                num_heads,
                x_padded.shape[3],
                b_padded.shape[-1],
            ),
            dtype=x_padded.dtype,
        )
    else:
        init_state = initial_states.astype(x_padded.dtype)

    d_skip_term = d_skip.astype(x_padded.dtype)

    def _combine(
        left: tuple[jax.Array, jax.Array],
        right: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        a_left, b_left = left
        a_right, b_right = right
        return (
            a_right * a_left,
            b_right + a_right[..., None, None] * b_left,
        )

    def _chunk_step(
        carry_state: jax.Array,
        chunk_inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        # Inputs already arrive as (chunk_size, batch, ...) thanks to the
        # single outer transpose above -- no per-iteration swap needed here.
        alpha_t, dt_t, b_t, c_t, x_t = chunk_inputs

        # dt_t: (t, b, h)   x_t: (t, b, h, p)   b_t: (t, b, n)
        # -> state_input_t: (t, b, h, p, n)
        # The head axis 'h' only appears in dt_t/x_t's subscripts, so
        # jnp.einsum broadcasts b_t across heads on the fly instead of us
        # ever materializing a (t, b, h, n)-shaped copy of b_t.
        state_input_t = jnp.einsum("tbh,tbhp,tbn->tbhpn", dt_t, x_t, b_t)

        alpha_prefix, state_prefix = jax.lax.associative_scan(
            _combine,
            (alpha_t, state_input_t),
            axis=0,
        )
        # c_t: (t, b, n), state_prefix: (t, b, h, p, n) -> (t, b, h, p)
        local_y = jnp.einsum("tbn,tbhpn->tbhp", c_t, state_prefix)
        carry_y = alpha_prefix[..., None] * jnp.einsum(
            "tbn,bhpn->tbhp",
            c_t,
            carry_state,
        )
        y_t = local_y + carry_y + d_skip_term[None, None, :, None] * x_t

        final_alpha = alpha_prefix[-1][..., None, None]
        final_state = state_prefix[-1] + final_alpha * carry_state
        return final_state, y_t

    final_state, y_chunks = jax.lax.scan(
        _chunk_step,
        init_state,
        (alpha_scan, dt_scan, b_scan, c_scan, x_scan),
    )

    # y_chunks is now (num_chunks, chunk_size, batch, heads, head_dim) since
    # the scan body no longer swaps back to (batch, chunk_size, ...) per
    # iteration -- permute straight to (batch, num_chunks, chunk_size, ...)
    # in one shot here instead.
    y = jnp.transpose(y_chunks, (2, 0, 1, 3, 4)).reshape(
        x_padded.shape[0], padded_len, num_heads, x_padded.shape[3]
    )
    if pad_size > 0:
        y = y[:, :seq_len, :, :]

    return y, final_state


class RMSNorm(nn.Module):
    hidden_size: int
    eps: float = 1e-6
    gate_residual: bool = False

    @nn.compact
    def __call__(
        self,
        hidden_states: jax.Array,
        residual: jax.Array | None = None,
    ) -> jax.Array:
        weight = self.param("weight", nn.initializers.ones, (self.hidden_size,))
        x = hidden_states.astype(jnp.float32)
        if residual is not None and self.gate_residual:
            x = x * nn.silu(residual.astype(jnp.float32))
        variance = jnp.mean(x**2, axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + self.eps) * weight
        return x.astype(hidden_states.dtype)


class DepthwiseConv1d(nn.Module):
    features: int
    kernel_size: int
    use_bias: bool = True

    @nn.compact
    def __call__(
        self,
        x: jax.Array,
        conv_state: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        cache_len = self.kernel_size - 1
        use_causal_pad = conv_state is None

        # On the no-cache path (this is what every full-sequence training
        # forward pass takes -- MambaLanguageModel discards the returned
        # conv/ssm state entirely), let the conv op apply the causal
        # zero-padding itself instead of building a separate padded copy of
        # `x` with jnp.pad first. Mathematically identical to
        # pad-then-VALID-conv, but XLA can fuse the padding into the
        # convolution instead of materializing an extra
        # (batch, seq_len + cache_len, channels) buffer.
        conv = nn.Conv(
            features=self.features,
            kernel_size=(self.kernel_size,),
            padding=[(cache_len, 0)] if use_causal_pad else "VALID",
            feature_group_count=self.features,
            use_bias=self.use_bias,
            name="conv",
        )

        if use_causal_pad:
            output = conv(x)
            if cache_len == 0:
                new_conv_state = jnp.zeros(
                    (x.shape[0], self.features, 0), dtype=x.dtype
                )
            elif x.shape[1] >= cache_len:
                # The cache is just the last cache_len steps of x itself --
                # padding only ever affects the *start* of the sequence, so
                # the tail doesn't need the padded array at all.
                new_conv_state = jnp.transpose(x[:, -cache_len:, :], (0, 2, 1))
            else:
                # Rare edge case: sequence shorter than the cache. Fall back
                # to materializing the padded tail explicitly.
                x_padded = jnp.pad(
                    x,
                    ((0, 0), (cache_len, 0), (0, 0)),
                    mode="constant",
                    constant_values=0.0,
                )
                new_conv_state = jnp.transpose(x_padded[:, -cache_len:, :], (0, 2, 1))
            return output, new_conv_state

        x_padded = jnp.concatenate([jnp.transpose(conv_state, (0, 2, 1)), x], axis=1)
        output = conv(x_padded)
        if cache_len > 0:
            new_conv_state = jnp.transpose(x_padded[:, -cache_len:, :], (0, 2, 1))
        else:
            new_conv_state = jnp.zeros((x.shape[0], self.features, 0), dtype=x.dtype)
        return output, new_conv_state


class Mamba2Mixer(nn.Module):
    hidden_size: int
    state_size: int = 16
    head_dim: int = 16
    chunk_size: int = 256
    expand: int = 2
    conv_kernel: int = 4
    use_bias: bool = False
    use_conv_bias: bool = True
    dt_min: float = 0.0
    dt_max: float = float("inf")
    hidden_act: str = "silu"

    @property
    def intermediate_size(self) -> int:
        return int(self.expand * self.hidden_size)

    @property
    def num_heads(self) -> int:
        return self.intermediate_size // self.head_dim

    def _activate(self, x: jax.Array) -> jax.Array:
        if self.hidden_act == "silu":
            return nn.silu(x)
        if self.hidden_act == "gelu":
            return nn.gelu(x)
        if self.hidden_act == "relu":
            return nn.relu(x)
        if self.hidden_act == "tanh":
            return jnp.tanh(x)
        raise RuntimeError(f"Unsupported hidden_act='{self.hidden_act}'")

    @nn.compact
    def __call__(
        self,
        hidden_states: jax.Array,
        conv_state: jax.Array | None = None,
        ssm_state: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        if self.intermediate_size % self.head_dim != 0:
            raise RuntimeError(
                "expand*hidden_size must be divisible by head_dim. "
                f"Got intermediate_size={self.intermediate_size}, "
                f"head_dim={self.head_dim}."
            )

        batch_size, seq_len, _ = hidden_states.shape
        proj_size = 2 * (self.intermediate_size + self.state_size) + self.num_heads
        zxbcdt = nn.Dense(
            proj_size,
            use_bias=self.use_bias,
            name="in_proj",
        )(hidden_states)

        d_mlp = (
            zxbcdt.shape[-1]
            - 2 * self.intermediate_size
            - 2 * self.state_size
            - self.num_heads
        ) // 2
        z0, x0, z, xbc, dt = jnp.split(
            zxbcdt,
            [
                d_mlp,
                2 * d_mlp,
                2 * d_mlp + self.intermediate_size,
                2 * d_mlp
                + self.intermediate_size
                + self.intermediate_size
                + 2 * self.state_size,
            ],
            axis=-1,
        )

        xbc, new_conv_state = DepthwiseConv1d(
            features=self.intermediate_size + 2 * self.state_size,
            kernel_size=self.conv_kernel,
            use_bias=self.use_conv_bias,
            name="conv1d",
        )(xbc, conv_state=conv_state)
        xbc = self._activate(xbc)
        x, b_t, c_t = jnp.split(
            xbc,
            [self.intermediate_size, self.intermediate_size + self.state_size],
            axis=-1,
        )

        a_log = self.param(
            "a_log",
            nn.initializers.uniform(scale=1.0),
            (self.num_heads,),
        )
        dt_bias = self.param(
            "dt_bias",
            _delta_bias_initializer(),
            (self.num_heads,),
        )
        d_skip = self.param("d", nn.initializers.ones, (self.num_heads,))
        a = -jnp.exp(a_log.astype(jnp.float32))

        # b_t / c_t are passed through as (batch, seq_len, state_size) --
        # NOT pre-broadcast to (batch, seq_len, num_heads, state_size).
        # _ssd_forward broadcasts them over heads implicitly via einsum,
        # which avoids materializing num_heads redundant copies of the same
        # data (previously done here via jnp.broadcast_to).
        y, new_ssm_state = _ssd_forward(
            x=x.reshape(batch_size, seq_len, self.num_heads, self.head_dim),
            dt=dt,
            a=a,
            b_mat=b_t,
            c_mat=c_t,
            chunk_size=self.chunk_size,
            d_skip=d_skip,
            dt_bias=dt_bias,
            dt_min=self.dt_min,
            dt_max=self.dt_max,
            initial_states=ssm_state,
        )
        y = y.reshape(batch_size, seq_len, self.intermediate_size)

        y = RMSNorm(
            self.intermediate_size,
            eps=1e-5,
            gate_residual=True,
            name="norm",
        )(y, residual=z)
        if d_mlp > 0:
            y = jnp.concatenate([self._activate(z0) * x0, y], axis=-1)

        out = nn.Dense(
            self.hidden_size,
            use_bias=self.use_bias,
            name="out_proj",
        )(y)
        return out, new_conv_state, new_ssm_state


class Mamba2Block(nn.Module):
    hidden_size: int
    state_size: int = 16
    head_dim: int = 16
    chunk_size: int = 256
    expand: int = 2
    conv_kernel: int = 4
    use_bias: bool = False
    use_conv_bias: bool = True
    layer_norm_epsilon: float = 1e-5
    residual_in_fp32: bool = True

    @nn.compact
    def __call__(
        self,
        hidden_states: jax.Array,
        conv_state: jax.Array | None = None,
        ssm_state: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        residual = hidden_states
        hs = RMSNorm(
            self.hidden_size,
            eps=self.layer_norm_epsilon,
            name="pre_norm",
        )(hidden_states.astype(jnp.float32))
        if self.residual_in_fp32:
            residual = residual.astype(jnp.float32)
        hs_out, new_conv_state, new_ssm_state = Mamba2Mixer(
            hidden_size=self.hidden_size,
            state_size=self.state_size,
            head_dim=self.head_dim,
            chunk_size=self.chunk_size,
            expand=self.expand,
            conv_kernel=self.conv_kernel,
            use_bias=self.use_bias,
            use_conv_bias=self.use_conv_bias,
            name="mixer",
        )(hs, conv_state=conv_state, ssm_state=ssm_state)
        return residual + hs_out, new_conv_state, new_ssm_state


class MambaLanguageModel(nn.Module):
    hidden_dim: int
    num_layers: int
    vocab_size: int
    embed_dim: int | None = None
    state_size: int = 16
    head_dim: int = 16
    chunk_size: int = 256
    expand: int = 2
    conv_kernel: int = 4
    use_bias: bool = False
    use_conv_bias: bool = True
    layer_norm_epsilon: float = 1e-5
    residual_in_fp32: bool = True

    def _model_dim(self) -> int:
        return self.hidden_dim if self.embed_dim is None else self.embed_dim

    def _intermediate_dim(self) -> int:
        return int(self.expand * self._model_dim())

    def _num_heads(self) -> int:
        if self._intermediate_dim() % self.head_dim != 0:
            raise RuntimeError(
                "expand*model_dim must be divisible by head_dim. "
                f"Got {self._intermediate_dim()} and head_dim={self.head_dim}."
            )
        return self._intermediate_dim() // self.head_dim

    def init_carry(self, batch_size: int) -> tuple[dict[str, jax.Array], ...]:
        cache_len = self.conv_kernel - 1
        conv_dim = self._intermediate_dim() + 2 * self.state_size
        num_heads = self._num_heads()
        carry: list[dict[str, jax.Array]] = []
        for _ in range(self.num_layers):
            carry.append(
                {
                    "conv_state": jnp.zeros((batch_size, conv_dim, cache_len)),
                    "ssm_state": jnp.zeros(
                        (batch_size, num_heads, self.head_dim, self.state_size)
                    ),
                }
            )
        return tuple(carry)

    @nn.compact
    def __call__(self, tokens: jax.Array) -> jax.Array:
        model_dim = self._model_dim()
        x = nn.Embed(num_embeddings=self.vocab_size, features=model_dim, name="embed")(
            tokens
        )

        for layer_idx in range(self.num_layers):
            x, _, _ = Mamba2Block(
                hidden_size=model_dim,
                state_size=self.state_size,
                head_dim=self.head_dim,
                chunk_size=self.chunk_size,
                expand=self.expand,
                conv_kernel=self.conv_kernel,
                use_bias=self.use_bias,
                use_conv_bias=self.use_conv_bias,
                layer_norm_epsilon=self.layer_norm_epsilon,
                residual_in_fp32=self.residual_in_fp32,
                name=f"mamba_{layer_idx}",
            )(x)

        x = RMSNorm(
            model_dim,
            eps=self.layer_norm_epsilon,
            name="final_norm",
        )(x)
        logits = nn.Dense(features=self.vocab_size, use_bias=False, name="lm_head")(x)
        return logits

    @nn.compact
    def step(
        self,
        carry: tuple[dict[str, jax.Array], ...],
        token: jax.Array,
    ) -> tuple[tuple[dict[str, jax.Array], ...], jax.Array]:
        model_dim = self._model_dim()
        x = nn.Embed(num_embeddings=self.vocab_size, features=model_dim, name="embed")(
            token
        )
        x = x[:, None, :]

        next_carry: list[dict[str, jax.Array]] = []
        for layer_idx in range(self.num_layers):
            x, new_conv_state, new_ssm_state = Mamba2Block(
                hidden_size=model_dim,
                state_size=self.state_size,
                head_dim=self.head_dim,
                chunk_size=self.chunk_size,
                expand=self.expand,
                conv_kernel=self.conv_kernel,
                use_bias=self.use_bias,
                use_conv_bias=self.use_conv_bias,
                layer_norm_epsilon=self.layer_norm_epsilon,
                residual_in_fp32=self.residual_in_fp32,
                name=f"mamba_{layer_idx}",
            )(
                x,
                conv_state=carry[layer_idx]["conv_state"],
                ssm_state=carry[layer_idx]["ssm_state"],
            )
            next_carry.append(
                {
                    "conv_state": new_conv_state,
                    "ssm_state": new_ssm_state,
                }
            )

        x = RMSNorm(
            model_dim,
            eps=self.layer_norm_epsilon,
            name="final_norm",
        )(x)
        logits = nn.Dense(features=self.vocab_size, use_bias=False, name="lm_head")(
            x[:, 0, :]
        )
        return tuple(next_carry), logits


def _verify_mamba_initialization(
    params: dict,
    *,
    model_dim: int,
    expand: int,
    state_size: int,
    head_dim: int,
) -> None:
    d_inner = int(model_dim * expand)
    num_heads = d_inner // head_dim
    mixer = params["mamba_0"]["mixer"]

    a = -jnp.exp(mixer["a_log"])
    if a.shape != (num_heads,):
        raise RuntimeError(f"A shape mismatch: {a.shape} != {(num_heads,)}")
    if not bool(jnp.all(a < 0.0)):
        raise RuntimeError("A must be strictly negative for stable dynamics")

    dt_bias = mixer["dt_bias"]
    if dt_bias.shape != (num_heads,):
        raise RuntimeError(
            f"Delta bias shape mismatch: {dt_bias.shape} != {(num_heads,)}"
        )
    dt_mean = float(jnp.mean(nn.softplus(dt_bias)))
    if not (0.003 <= dt_mean <= 0.03):
        raise RuntimeError(
            "Delta mean outside expected range near geometric midpoint: "
            f"mean={dt_mean:.6f}"
        )

    conv_kernel = mixer["conv1d"]["conv"]["kernel"]
    if conv_kernel.shape[-1] != d_inner + 2 * state_size:
        raise RuntimeError(
            "Conv projection mismatch: "
            f"kernel={conv_kernel.shape}, expected channels={d_inner + 2 * state_size}"
        )


def _run_unit_tests() -> None:
    key = jax.random.PRNGKey(0)
    batch, seq_len = 3, 12
    vocab_size = 101
    model = MambaLanguageModel(
        hidden_dim=16,
        num_layers=2,
        vocab_size=vocab_size,
        state_size=8,
        head_dim=8,
        chunk_size=4,
        expand=2,
        conv_kernel=4,
    )

    tokens = jax.random.randint(key, (batch, seq_len), minval=0, maxval=vocab_size)
    vars_ = model.init(key, tokens)
    _verify_mamba_initialization(
        vars_["params"],
        model_dim=16,
        expand=2,
        state_size=8,
        head_dim=8,
    )

    logits = model.apply(vars_, tokens)
    assert logits.shape == (batch, seq_len, vocab_size)

    t = 7
    tokens_changed = tokens.at[:, t].set((tokens[:, t] + 1) % vocab_size)
    logits_ref = model.apply(vars_, tokens)
    logits_changed = model.apply(vars_, tokens_changed)
    max_pre_t_diff = jnp.max(jnp.abs(logits_ref[:, :t, :] - logits_changed[:, :t, :]))
    assert float(max_pre_t_diff) < 1e-6, f"Causality violated: {float(max_pre_t_diff)}"

    carry = model.init_carry(batch)
    step_logits = []
    for i in range(seq_len):
        carry, logits_i = model.apply(vars_, carry, tokens[:, i], method=model.step)
        step_logits.append(logits_i)
    stacked = jnp.stack(step_logits, axis=1)
    max_diff = jnp.max(jnp.abs(logits - stacked))
    assert float(max_diff) < 3e-3, f"Parallel/step mismatch: {float(max_diff)}"

    print("Mamba (Bonsai-style) unit checks passed")


if __name__ == "__main__":
    _run_unit_tests()
