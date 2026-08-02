from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


class _GateParams(nn.Module):
    """Owns one flax-LSTMCell-compatible ``{kernel, bias}`` pair.

    Exists only so the fused layer below can declare parameters at exactly the
    paths ``nn.LSTMCell`` uses (``ii/kernel``, ``hi/kernel``, ``hi/bias``, ...),
    which keeps checkpoints written by the previous implementation loadable.
    """

    in_features: int
    out_features: int
    kernel_init: nn.initializers.Initializer
    use_bias: bool

    @nn.compact
    def __call__(self) -> tuple[jax.Array, jax.Array | None]:
        kernel = self.param(
            "kernel", self.kernel_init, (self.in_features, self.out_features)
        )
        bias = (
            self.param("bias", nn.initializers.zeros_init(), (self.out_features,))
            if self.use_bias
            else None
        )
        return kernel, bias


# Gate order is fixed by flax's LSTMCell parameter names; the fused kernels
# below concatenate in this order and jnp.split unpacks in the same order.
_INPUT_GATES = ("ii", "if", "ig", "io")
_HIDDEN_GATES = ("hi", "hf", "hg", "ho")


class FusedLSTMLayer(nn.Module):
    """``nn.RNN(nn.LSTMCell(H))`` with identical parameters and identical math.

    Two changes, both exact:

    1. The input projections W_ii x, W_if x, W_ig x, W_io x do not depend on the
       carry, so they are computed for every timestep in a single GEMM *before*
       the scan instead of four small GEMMs inside it. Hoisting a loop-invariant
       computation out of the loop -- same arithmetic, done once.
    2. The four input kernels are concatenated into one (D, 4H) matrix and the
       four hidden kernels into one (H, 4H), so each step issues one matmul
       instead of eight. Concatenating along the output axis and splitting after
       is exactly equivalent: every output element is the same dot product of
       the same operands.

    At seq_len 8192 the recurrence issues 8192 sequential steps, so per-step
    kernel-launch count and matmul size dominate; this turns 8 tiny launches per
    step into 1 larger one plus a single large GEMM outside the loop.
    """

    hidden_dim: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        hidden_dim = self.hidden_dim
        in_features = x.shape[-1]
        batch_size = x.shape[0]

        input_kernels = [
            _GateParams(
                in_features,
                hidden_dim,
                nn.initializers.lecun_normal(),
                False,
                name=name,
            )()[0]
            for name in _INPUT_GATES
        ]
        hidden_pairs = [
            _GateParams(
                hidden_dim,
                hidden_dim,
                nn.initializers.orthogonal(),
                True,
                name=name,
            )()
            for name in _HIDDEN_GATES
        ]

        w_input = jnp.concatenate(input_kernels, axis=1)
        w_hidden = jnp.concatenate([kernel for kernel, _ in hidden_pairs], axis=1)
        b_hidden = jnp.concatenate([bias for _, bias in hidden_pairs])

        # One GEMM for every timestep at once, hoisted out of the recurrence.
        gates_from_input = x @ w_input

        def step(carry, gate_x):
            c, h = carry
            gates = gate_x + h @ w_hidden + b_hidden
            i, f, g, o = jnp.split(gates, 4, axis=-1)
            new_c = nn.sigmoid(f) * c + nn.sigmoid(i) * jnp.tanh(g)
            new_h = nn.sigmoid(o) * jnp.tanh(new_c)
            return (new_c, new_h), new_h

        zeros = jnp.zeros((batch_size, hidden_dim), dtype=gates_from_input.dtype)
        _, outputs = jax.lax.scan(
            step, (zeros, zeros), jnp.swapaxes(gates_from_input, 0, 1)
        )
        return jnp.swapaxes(outputs, 0, 1)


class LSTMLanguageModel(nn.Module):
    hidden_dim: int
    num_layers: int
    vocab_size: int
    embed_dim: int | None = None

    def init_carry(self, batch_size: int) -> tuple:
        """Initialize LSTM carry for each layer."""
        carry = []
        for _ in range(self.num_layers):
            c = jnp.zeros((batch_size, self.hidden_dim))
            h = jnp.zeros((batch_size, self.hidden_dim))
            carry.append((c, h))
        return tuple(carry)

    @nn.compact
    def __call__(self, tokens: jax.Array) -> jax.Array:
        embed_dim = self.hidden_dim if self.embed_dim is None else self.embed_dim

        x = nn.Embed(num_embeddings=self.vocab_size, features=embed_dim, name="embed")(
            tokens
        )
        # x shape: (batch, seq_len, embed_dim)

        for layer_idx in range(self.num_layers):
            # Named LSTMCell_<i> to match the parameter paths nn.RNN produced,
            # so checkpoints from before the fused rewrite still restore.
            x = FusedLSTMLayer(self.hidden_dim, name=f"LSTMCell_{layer_idx}")(x)
            # x shape: (batch, seq_len, hidden_dim)

        logits = nn.Dense(features=self.vocab_size, name="lm_head")(x)
        return logits

    @nn.compact
    def step(self, carry: tuple, token: jax.Array) -> tuple:
        """Single LSTM step for autoregressive sampling.

        Args:
            carry: LSTM carry (tuple of (c, h) per layer)
            token: (batch,) integer array of a single token

        Returns:
            (new_carry, logits) where logits is (batch, vocab_size)
        """
        embed_dim = self.hidden_dim if self.embed_dim is None else self.embed_dim

        x = nn.Embed(num_embeddings=self.vocab_size, features=embed_dim, name="embed")(
            token
        )

        class _StepCell(nn.Module):
            hidden_dim: int

            @nn.compact
            def __call__(self, cell_carry: tuple, inputs: jax.Array) -> tuple:
                lstm_cell = nn.LSTMCell(self.hidden_dim, name="cell")
                return lstm_cell(cell_carry, inputs)

        new_carry = []
        for layer_idx in range(self.num_layers):
            step_cell = _StepCell(self.hidden_dim, name=f"rnn_{layer_idx}")
            layer_carry = carry[layer_idx] if carry else None
            new_layer_carry, x = step_cell(layer_carry, x)
            new_carry.append(new_layer_carry)

        logits = nn.Dense(features=self.vocab_size, name="lm_head")(x)
        return (tuple(new_carry), logits)
