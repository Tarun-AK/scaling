from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp


def _rotate_half(x: jax.Array) -> jax.Array:
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated = jnp.stack((-x_odd, x_even), axis=-1)
    return rotated.reshape(x.shape)


def _apply_rope(x: jax.Array, theta: float) -> jax.Array:
    _, seq_len, _, head_dim = x.shape
    if head_dim % 2 != 0:
        raise RuntimeError(f"RoPE requires even head_dim, got {head_dim}")

    inv_freq = 1.0 / (
        jnp.asarray(theta, dtype=jnp.float32)
        ** (jnp.arange(0, head_dim, 2, dtype=jnp.float32) / float(head_dim))
    )
    positions = jnp.arange(seq_len, dtype=jnp.float32)
    freqs = positions[:, None] * inv_freq[None, :]
    cos = jnp.repeat(jnp.cos(freqs), 2, axis=-1)[None, :, None, :]
    sin = jnp.repeat(jnp.sin(freqs), 2, axis=-1)[None, :, None, :]

    x_f32 = x.astype(jnp.float32)
    return (x_f32 * cos + _rotate_half(x_f32) * sin).astype(x.dtype)


class RMSNorm(nn.Module):
    hidden_size: int
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        weight = self.param("weight", nn.initializers.ones, (self.hidden_size,))
        x_f32 = x.astype(jnp.float32)
        variance = jnp.mean(x_f32**2, axis=-1, keepdims=True)
        x_norm = x_f32 * jax.lax.rsqrt(variance + self.eps)
        return (x_norm * weight).astype(x.dtype)


class CausalSelfAttention(nn.Module):
    hidden_dim: int
    num_heads: int
    num_kv_heads: int
    rope_theta: float
    # None -> XLA path (materializes the seq x seq score matrix, as before).
    # "cudnn" -> fused flash kernel, which never writes that matrix to HBM.
    # At seq_len 8192 the materialized scores are ~2.1GB per layer per sequence
    # in fp32, which makes training memory-bandwidth-bound rather than
    # compute-bound; "cudnn" is what removes that wall.
    attention_impl: str | None = None

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        head_dim = self.hidden_dim // self.num_heads
        if self.hidden_dim % self.num_heads != 0:
            raise RuntimeError(
                "hidden_dim must be divisible by num_heads. "
                f"Got hidden_dim={self.hidden_dim}, num_heads={self.num_heads}"
            )
        if self.num_heads % self.num_kv_heads != 0:
            raise RuntimeError(
                "num_heads must be divisible by num_kv_heads. "
                f"Got num_heads={self.num_heads}, num_kv_heads={self.num_kv_heads}"
            )

        q = nn.Dense(self.hidden_dim, use_bias=False, name="q_proj")(x)
        kv_dim = self.num_kv_heads * head_dim
        k = nn.Dense(kv_dim, use_bias=False, name="k_proj")(x)
        v = nn.Dense(kv_dim, use_bias=False, name="v_proj")(x)

        batch_size, seq_len, _ = x.shape
        q = q.reshape(batch_size, seq_len, self.num_heads, head_dim)
        k = k.reshape(batch_size, seq_len, self.num_kv_heads, head_dim)
        v = v.reshape(batch_size, seq_len, self.num_kv_heads, head_dim)

        q = _apply_rope(q, self.rope_theta)
        k = _apply_rope(k, self.rope_theta)

        if self.num_kv_heads != self.num_heads:
            repeat = self.num_heads // self.num_kv_heads
            k = jnp.repeat(k, repeat, axis=2)
            v = jnp.repeat(v, repeat, axis=2)

        scale = head_dim**-0.5
        # cuDNN's fused kernel accepts only fp16/bf16 -- passing fp32 raises
        # rather than silently falling back, so cast just the attention core and
        # return to the surrounding dtype. Projections stay in x.dtype.
        compute_dtype = jnp.bfloat16 if self.attention_impl == "cudnn" else x.dtype
        attn_out = jax.nn.dot_product_attention(
            q.astype(compute_dtype),
            k.astype(compute_dtype),
            v.astype(compute_dtype),
            scale=scale,
            is_causal=True,
            implementation=self.attention_impl,
        ).astype(x.dtype)
        attn_out = attn_out.reshape(batch_size, seq_len, self.hidden_dim)
        return nn.Dense(self.hidden_dim, use_bias=False, name="o_proj")(attn_out)


class SwiGLUMLP(nn.Module):
    hidden_dim: int
    ffn_mult: float

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        ffn_dim = max(1, int(self.ffn_mult * self.hidden_dim))
        gate = nn.Dense(ffn_dim, use_bias=False, name="gate_proj")(x)
        up = nn.Dense(ffn_dim, use_bias=False, name="up_proj")(x)
        act = nn.silu(gate) * up
        return nn.Dense(self.hidden_dim, use_bias=False, name="down_proj")(act)


class TransformerBlock(nn.Module):
    hidden_dim: int
    num_heads: int
    num_kv_heads: int
    ffn_mult: float
    rope_theta: float
    layer_norm_epsilon: float = 1e-6
    attention_impl: str | None = None

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        h = x + CausalSelfAttention(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_kv_heads=self.num_kv_heads,
            rope_theta=self.rope_theta,
            attention_impl=self.attention_impl,
            name="self_attn",
        )(RMSNorm(self.hidden_dim, eps=self.layer_norm_epsilon, name="attn_norm")(x))
        out = h + SwiGLUMLP(
            hidden_dim=self.hidden_dim,
            ffn_mult=self.ffn_mult,
            name="mlp",
        )(RMSNorm(self.hidden_dim, eps=self.layer_norm_epsilon, name="mlp_norm")(h))
        return out


class TransformerLanguageModel(nn.Module):
    hidden_dim: int
    num_layers: int
    vocab_size: int
    num_heads: int
    num_kv_heads: int | None = None
    ffn_mult: float = 8.0 / 3.0
    rope_theta: float = 1_000_000.0
    layer_norm_epsilon: float = 1e-6
    tie_word_embeddings: bool = True
    attention_impl: str | None = None

    def init_carry(self, batch_size: int) -> dict[str, jax.Array]:
        return {"tokens": jnp.zeros((batch_size, 0), dtype=jnp.int32)}

    def _effective_num_kv_heads(self) -> int:
        return self.num_heads if self.num_kv_heads is None else int(self.num_kv_heads)

    @nn.compact
    def _forward_tokens(self, tokens: jax.Array) -> jax.Array:
        embed = nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.hidden_dim,
            name="embed",
        )
        x = embed(tokens)
        num_kv_heads = self._effective_num_kv_heads()

        for layer_idx in range(self.num_layers):
            x = TransformerBlock(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                num_kv_heads=num_kv_heads,
                ffn_mult=self.ffn_mult,
                rope_theta=self.rope_theta,
                layer_norm_epsilon=self.layer_norm_epsilon,
                attention_impl=self.attention_impl,
                name=f"layer_{layer_idx}",
            )(x)

        x = RMSNorm(
            self.hidden_dim,
            eps=self.layer_norm_epsilon,
            name="final_norm",
        )(x)
        if self.tie_word_embeddings:
            return embed.attend(x)
        return nn.Dense(self.vocab_size, use_bias=False, name="lm_head")(x)

    @nn.compact
    def __call__(self, tokens: jax.Array) -> jax.Array:
        return self._forward_tokens(tokens)

    @nn.compact
    def step(
        self,
        carry: dict[str, jax.Array],
        token: jax.Array,
    ) -> tuple[dict[str, jax.Array], jax.Array]:
        prev_tokens = carry["tokens"]
        all_tokens = jnp.concatenate([prev_tokens, token[:, None]], axis=1)
        logits = self._forward_tokens(all_tokens)
        return {"tokens": all_tokens}, logits[:, -1, :]
