"""
OpenMythos-MLX — Recurrent-Depth Transformer for Apple Silicon

Port of OpenMythos (PyTorch) to Apple MLX framework.
Architecture: Prelude → Recurrent Block (looped T times) → Coda

Key components:
- MLA (Multi-Latent Attention) or GQA (Grouped Query Attention)
- MoE FFN (DeepSeek-style Mixture of Experts)
- ACT (Adaptive Computation Time) halting
- LTI-stable input injection (spectral radius < 1 guaranteed)
- LoRA depth adaptation per loop iteration
- RoPE positional embeddings

Original: https://github.com/DeadByDawn101/OpenMythos
MLX Port: https://github.com/DeadByDawn101/OpenMythos-MLX
Author: RavenX AI / DeadByDawn101
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import math

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class MythosConfig:
    """Hyperparameter configuration for OpenMythos-MLX."""
    vocab_size: int = 32000
    dim: int = 2048
    n_heads: int = 16
    n_kv_heads: int = 4
    max_seq_len: int = 4096
    max_loop_iters: int = 16
    prelude_layers: int = 2
    coda_layers: int = 2
    attn_type: str = "mla"  # "gqa" or "mla"
    # MLA params
    kv_lora_rank: int = 512
    q_lora_rank: int = 1536
    qk_rope_head_dim: int = 64
    qk_nope_head_dim: int = 128
    v_head_dim: int = 128
    # MoE
    n_experts: int = 64
    n_shared_experts: int = 2
    n_experts_per_tok: int = 4
    expert_dim: int = 512
    # ACT
    act_threshold: float = 0.99
    # RoPE
    rope_theta: float = 500000.0
    # LoRA
    lora_rank: int = 16
    max_output_tokens: int = 4096
    dropout: float = 0.0


# ---------------------------------------------------------------------------
# RoPE (real-valued, no complex numbers in MLX)
# ---------------------------------------------------------------------------

def precompute_rope_freqs(dim: int, max_len: int, theta: float = 500000.0) -> Tuple[mx.array, mx.array]:
    """Precompute sin/cos for RoPE. Returns (cos, sin) each of shape (max_len, dim//2)."""
    freqs = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
    t = mx.arange(max_len, dtype=mx.float32)
    angles = mx.outer(t, freqs)
    return mx.cos(angles), mx.sin(angles)


def apply_rope(x: mx.array, cos_freqs: mx.array, sin_freqs: mx.array) -> mx.array:
    """Apply RoPE using real-valued sin/cos rotation.
    x: (B, T, H, head_dim) — head_dim must be even
    cos_freqs, sin_freqs: (T, head_dim//2) sliced to match positions
    """
    B, T, H, D = x.shape
    half = D // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    cos_f = cos_freqs[:T][None, :, None, :]  # (1, T, 1, half)
    sin_f = sin_freqs[:T][None, :, None, :]
    out1 = x1 * cos_f - x2 * sin_f
    out2 = x1 * sin_f + x2 * cos_f
    return mx.concatenate([out1, out2], axis=-1)


# ---------------------------------------------------------------------------
# Loop index embedding
# ---------------------------------------------------------------------------

def loop_index_embedding(h: mx.array, t: int, embed_dim: int) -> mx.array:
    """Add sinusoidal loop-index signal to first embed_dim channels."""
    D = embed_dim
    half = D // 2
    pos = mx.array([t], dtype=mx.float32)
    freqs = 1.0 / (10000.0 ** (mx.arange(0, half, dtype=mx.float32) / half))
    angles = pos * freqs
    sin_vals = mx.sin(angles)
    cos_vals = mx.cos(angles)
    # Build signal: [sin0, cos0, sin1, cos1, ...]
    signal = mx.zeros((D,))
    for i in range(min(half, len(sin_vals))):
        signal = signal * 1.0  # force eval
    # Simple approach: concatenate sin and cos
    signal = mx.concatenate([sin_vals, cos_vals])[:D]
    # Pad to match h's last dim and add
    if h.shape[-1] > D:
        pad = mx.zeros((h.shape[-1] - D,))
        signal = mx.concatenate([signal, pad])
    return h + signal


# ---------------------------------------------------------------------------
# GQA (Grouped Query Attention)
# ---------------------------------------------------------------------------

class GQAttention(nn.Module):
    """Grouped Query Attention with KV cache support."""

    def __init__(self, cfg: MythosConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.dim // cfg.n_heads
        self.n_rep = cfg.n_heads // cfg.n_kv_heads

        self.wq = nn.Linear(cfg.dim, cfg.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * self.head_dim, cfg.dim, bias=False)

    def __call__(self, x, cos_freqs, sin_freqs, mask=None, kv_cache=None, cache_key=None):
        B, T, _ = x.shape

        q = self.wq(x).reshape(B, T, self.n_heads, self.head_dim)
        k = self.wk(x).reshape(B, T, self.n_kv_heads, self.head_dim)
        v = self.wv(x).reshape(B, T, self.n_kv_heads, self.head_dim)

        q = apply_rope(q, cos_freqs, sin_freqs)
        k = apply_rope(k, cos_freqs, sin_freqs)

        # KV cache
        if kv_cache is not None and cache_key is not None:
            if cache_key in kv_cache:
                pk, pv = kv_cache[cache_key]
                k = mx.concatenate([pk, k], axis=1)
                v = mx.concatenate([pv, v], axis=1)
            kv_cache[cache_key] = (k, v)

        # Expand KV heads for GQA
        if self.n_rep > 1:
            k = mx.repeat(k, self.n_rep, axis=2)
            v = mx.repeat(v, self.n_rep, axis=2)

        # Transpose to (B, H, T, D)
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        scale = 1.0 / math.sqrt(self.head_dim)
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale

        if mask is not None:
            scores = scores + mask

        attn = mx.softmax(scores, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.wo(out)


# ---------------------------------------------------------------------------
# MLA (Multi-Latent Attention) — DeepSeek style
# ---------------------------------------------------------------------------

class MLAttention(nn.Module):
    """Multi-Latent Attention: compressed KV cache via learned low-rank projection."""

    def __init__(self, cfg: MythosConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.kv_lora_rank = cfg.kv_lora_rank
        self.qk_rope_dim = cfg.qk_rope_head_dim
        self.qk_nope_dim = cfg.qk_nope_head_dim
        self.v_head_dim = cfg.v_head_dim
        self.q_head_dim = self.qk_nope_dim + self.qk_rope_dim

        # Q path
        self.wq_compress = nn.Linear(cfg.dim, cfg.q_lora_rank, bias=False)
        self.wq_norm = nn.RMSNorm(cfg.q_lora_rank)
        self.wq_expand = nn.Linear(cfg.q_lora_rank, self.n_heads * self.q_head_dim, bias=False)

        # KV compressed path
        self.wkv_compress = nn.Linear(cfg.dim, self.kv_lora_rank, bias=False)
        self.wkv_norm = nn.RMSNorm(self.kv_lora_rank)
        self.wk_expand = nn.Linear(self.kv_lora_rank, self.n_heads * self.qk_nope_dim, bias=False)
        self.wv_expand = nn.Linear(self.kv_lora_rank, self.n_heads * self.v_head_dim, bias=False)

        # Decoupled RoPE key projection
        self.wk_rope = nn.Linear(cfg.dim, self.n_heads * self.qk_rope_dim, bias=False)

        # Output projection
        self.wo = nn.Linear(self.n_heads * self.v_head_dim, cfg.dim, bias=False)

    def __call__(self, x, cos_freqs, sin_freqs, mask=None, kv_cache=None, cache_key=None):
        B, T, _ = x.shape

        # Q: compress → norm → expand → split nope/rope
        q_c = self.wq_norm(self.wq_compress(x))
        q = self.wq_expand(q_c).reshape(B, T, self.n_heads, self.q_head_dim)
        q_nope = q[..., :self.qk_nope_dim]
        q_rope = q[..., self.qk_nope_dim:]
        q_rope = apply_rope(q_rope, cos_freqs, sin_freqs)

        # KV: compress → norm → expand
        kv_c = self.wkv_norm(self.wkv_compress(x))
        k_nope = self.wk_expand(kv_c).reshape(B, T, self.n_heads, self.qk_nope_dim)
        v = self.wv_expand(kv_c).reshape(B, T, self.n_heads, self.v_head_dim)

        # Decoupled RoPE keys
        k_rope = self.wk_rope(x).reshape(B, T, self.n_heads, self.qk_rope_dim)
        k_rope = apply_rope(k_rope, cos_freqs, sin_freqs)

        # Concatenate key components
        k = mx.concatenate([k_nope, k_rope], axis=-1)
        q = mx.concatenate([q_nope, q_rope], axis=-1)

        # KV cache
        if kv_cache is not None and cache_key is not None:
            if cache_key in kv_cache:
                pk, pv = kv_cache[cache_key]
                k = mx.concatenate([pk, k], axis=1)
                v = mx.concatenate([pv, v], axis=1)
            kv_cache[cache_key] = (k, v)

        # Attention
        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)

        scale = 1.0 / math.sqrt(self.q_head_dim)
        scores = (q @ k.transpose(0, 1, 3, 2)) * scale

        if mask is not None:
            scores = scores + mask

        attn = mx.softmax(scores, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, -1)
        return self.wo(out)


# ---------------------------------------------------------------------------
# Expert + MoE FFN
# ---------------------------------------------------------------------------

class Expert(nn.Module):
    """Single SwiGLU expert."""

    def __init__(self, dim: int, expert_dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, expert_dim, bias=False)
        self.up = nn.Linear(dim, expert_dim, bias=False)
        self.down = nn.Linear(expert_dim, dim, bias=False)

    def __call__(self, x):
        return self.down(nn.silu(self.gate(x)) * self.up(x))


class MoEFFN(nn.Module):
    """Mixture of Experts FFN with shared + routed experts."""

    def __init__(self, cfg: MythosConfig):
        super().__init__()
        self.n_experts = cfg.n_experts
        self.n_shared = cfg.n_shared_experts
        self.top_k = cfg.n_experts_per_tok

        self.router = nn.Linear(cfg.dim, cfg.n_experts, bias=False)
        self.experts = [Expert(cfg.dim, cfg.expert_dim) for _ in range(cfg.n_experts)]
        self.shared = [Expert(cfg.dim, cfg.expert_dim) for _ in range(self.n_shared)]

    def __call__(self, x):
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)

        # Shared experts (always active)
        shared_out = sum(expert(x_flat) for expert in self.shared)

        # Router scores
        scores = mx.softmax(self.router(x_flat), axis=-1)
        top_k_indices = mx.argpartition(-scores, kth=self.top_k, axis=-1)[..., :self.top_k]
        top_k_scores = mx.take_along_axis(scores, top_k_indices, axis=-1)
        top_k_scores = top_k_scores / (top_k_scores.sum(axis=-1, keepdims=True) + 1e-8)

        # Routed experts
        routed_out = mx.zeros_like(x_flat)
        for k in range(self.top_k):
            idx = top_k_indices[:, k]
            weight = top_k_scores[:, k:k+1]
            # Process each expert
            for e_idx in range(self.n_experts):
                mask = (idx == e_idx)
                if mx.any(mask):
                    expert_input = x_flat * mask[:, None].astype(x_flat.dtype)
                    routed_out = routed_out + weight * self.experts[e_idx](expert_input) * mask[:, None].astype(x_flat.dtype)

        out = (shared_out + routed_out).reshape(B, T, D)
        return out


# ---------------------------------------------------------------------------
# LoRA Adapter
# ---------------------------------------------------------------------------

class LoRAAdapter(nn.Module):
    """Depth-wise LoRA: different B matrices per loop iteration."""

    def __init__(self, dim: int, rank: int, max_iters: int):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)  # A matrix
        self.ups = [nn.Linear(rank, dim, bias=False) for _ in range(max_iters)]  # B matrices
        self.scale = 0.01

    def __call__(self, x, loop_idx: int):
        up = self.ups[min(loop_idx, len(self.ups) - 1)]
        return up(self.down(x)) * self.scale


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """Standard transformer block with configurable attention and optional MoE."""

    def __init__(self, cfg: MythosConfig, use_moe: bool = False):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.dim)
        self.norm2 = nn.RMSNorm(cfg.dim)

        if cfg.attn_type == "mla":
            self.attn = MLAttention(cfg)
        else:
            self.attn = GQAttention(cfg)

        if use_moe:
            self.ffn = MoEFFN(cfg)
        else:
            self.ffn = Expert(cfg.dim, cfg.dim * 4)

    def __call__(self, x, cos_freqs, sin_freqs, mask=None, kv_cache=None, cache_key=None):
        x = x + self.attn(self.norm1(x), cos_freqs, sin_freqs, mask, kv_cache,
                          f"{cache_key}_attn" if cache_key else None)
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# LTI Injection (guaranteed stable: spectral radius < 1)
# ---------------------------------------------------------------------------

class LTIInjection(nn.Module):
    """Stable input injection: h_{t+1} = A·h_t + B·e + transformer_out.
    Guarantees spectral radius < 1 via ZOH discretization."""

    def __init__(self, dim: int):
        super().__init__()
        # Store as nn.Linear trick to make them proper parameters
        self._log_A = mx.zeros((dim,))
        self._log_dt = mx.zeros((1,))
        self._B = mx.ones((dim,)) * 0.1
        self.dim = dim

    def get_A(self):
        # Clamp tightly to prevent any overflow
        combined = mx.clip(self._log_dt + self._log_A, -10, 2)
        return mx.exp(-mx.exp(combined))

    def __call__(self, h, e, transformer_out):
        A = self.get_A()
        # Scale down contributions to prevent explosion
        return A * h + self._B * e * 0.1 + transformer_out


# ---------------------------------------------------------------------------
# ACT Halting
# ---------------------------------------------------------------------------

class ACTHalting(nn.Module):
    """Adaptive Computation Time: per-position halting probability."""

    def __init__(self, dim: int):
        super().__init__()
        self.halt = nn.Linear(dim, 1)

    def __call__(self, h):
        return mx.sigmoid(self.halt(h)).squeeze(-1)


# ---------------------------------------------------------------------------
# Recurrent Block
# ---------------------------------------------------------------------------

class RecurrentBlock(nn.Module):
    """Core recurrent block — single TransformerBlock looped T times with ACT."""

    def __init__(self, cfg: MythosConfig):
        super().__init__()
        self.cfg = cfg
        self.block = TransformerBlock(cfg, use_moe=True)
        self.injection = LTIInjection(cfg.dim)
        self.act = ACTHalting(cfg.dim)
        self.lora = LoRAAdapter(cfg.dim, cfg.lora_rank, cfg.max_loop_iters)
        self.norm = nn.RMSNorm(cfg.dim)
        self.loop_dim = cfg.dim // 8

    def __call__(self, h, e, cos_freqs, sin_freqs, mask=None, n_loops=None, kv_cache=None):
        n_loops = n_loops or self.cfg.max_loop_iters
        B, T, D = h.shape

        halted = mx.zeros((B, T), dtype=mx.bool_)
        cumulative_p = mx.zeros((B, T))
        h_out = mx.zeros_like(h)

        for t in range(n_loops):
            h_loop = loop_index_embedding(h, t, self.loop_dim)
            combined = self.norm(h_loop + e)
            cache_key = f"recurrent_loop_{t}"
            trans_out = self.block(combined, cos_freqs, sin_freqs, mask, kv_cache, cache_key)
            trans_out = trans_out + self.lora(trans_out, t)
            h = self.injection(h, e, trans_out)

            p = self.act(h)
            still_running = ~halted

            remainder = mx.clip(1.0 - cumulative_p, a_min=0, a_max=None)
            weight = mx.where(
                cumulative_p + p >= self.cfg.act_threshold,
                remainder,
                p,
            )
            weight = weight * still_running.astype(mx.float32)
            h_out = h_out + mx.expand_dims(weight, -1) * h

            cumulative_p = cumulative_p + p * still_running.astype(mx.float32)
            halted = halted | (cumulative_p >= self.cfg.act_threshold)

            if mx.all(halted) and kv_cache is None:
                break

        return h_out


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class OpenMythos(nn.Module):
    """OpenMythos-MLX — Recurrent-Depth Transformer.

    Prelude → Recurrent Block (looped T times) → Coda → Logits

    Same weights, more loops → deeper reasoning, no parameter growth.
    """

    def __init__(self, cfg: MythosConfig):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)

        # Precompute RoPE frequencies
        head_dim = cfg.dim // cfg.n_heads
        self.cos_freqs, self.sin_freqs = precompute_rope_freqs(
            head_dim, cfg.max_seq_len, cfg.rope_theta
        )
        self.cos_freqs_mla, self.sin_freqs_mla = precompute_rope_freqs(
            cfg.qk_rope_head_dim, cfg.max_seq_len, cfg.rope_theta
        )

        self.prelude = [TransformerBlock(cfg, use_moe=False) for _ in range(cfg.prelude_layers)]
        self.recurrent = RecurrentBlock(cfg)
        self.coda = [TransformerBlock(cfg, use_moe=False) for _ in range(cfg.coda_layers)]

        self.norm = nn.RMSNorm(cfg.dim)
        self.head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        # Initialize weights with N(0, 0.02) like GPT-2
        self._init_weights()

    def _init_weights(self):
        """Initialize all Linear and Embedding weights with N(0, 0.02)."""
        def init_fn(module):
            if isinstance(module, nn.Linear):
                module.weight = mx.random.normal(module.weight.shape) * 0.02
            elif isinstance(module, nn.Embedding):
                module.weight = mx.random.normal(module.weight.shape) * 0.02
        self.apply_to_modules(init_fn)

    def apply_to_modules(self, fn):
        """Apply function to all sub-modules."""
        fn(self)
        for child in self.children().values():
            if isinstance(child, nn.Module):
                fn(child)
            elif isinstance(child, list):
                for item in child:
                    if isinstance(item, nn.Module):
                        fn(item)

    def __call__(self, input_ids, n_loops=None, kv_cache=None, start_pos=0):
        B, T = input_ids.shape

        x = self.embed(input_ids)

        if self.cfg.attn_type == "mla":
            cos_f = self.cos_freqs_mla[start_pos:start_pos + T]
            sin_f = self.sin_freqs_mla[start_pos:start_pos + T]
        else:
            cos_f = self.cos_freqs[start_pos:start_pos + T]
            sin_f = self.sin_freqs[start_pos:start_pos + T]

        # Causal mask
        if T > 1:
            mask = mx.full((1, 1, T, T), float('-inf'))
            mask = mx.triu(mask, k=1)
        else:
            mask = None

        # Prelude
        for i, layer in enumerate(self.prelude):
            x = layer(x, cos_f, sin_f, mask, kv_cache, cache_key=f"prelude_{i}")

        # Recurrent block
        e = x  # encoded input frozen for injection
        x = self.recurrent(x, e, cos_f, sin_f, mask, n_loops, kv_cache)

        # Coda
        for i, layer in enumerate(self.coda):
            x = layer(x, cos_f, sin_f, mask, kv_cache, cache_key=f"coda_{i}")

        return self.head(self.norm(x))

    def generate(self, input_ids, max_new_tokens=64, n_loops=8, temperature=1.0, top_k=50):
        """Autoregressive generation with KV caching."""
        kv_cache = {}
        prompt_len = input_ids.shape[1]

        for step in range(max_new_tokens):
            if step == 0:
                cur_ids = input_ids
                start_pos = 0
            else:
                cur_ids = input_ids[:, -1:]
                start_pos = prompt_len + step - 1

            logits = self(cur_ids, n_loops=n_loops, kv_cache=kv_cache, start_pos=start_pos)
            logits = logits[:, -1, :] / temperature

            if top_k > 0:
                top_vals = mx.sort(logits, axis=-1)[:, -top_k:]
                threshold = top_vals[:, 0:1]
                logits = mx.where(logits < threshold, float('-inf'), logits)

            probs = mx.softmax(logits, axis=-1)
            next_tok = mx.random.categorical(probs)[:, None]
            input_ids = mx.concatenate([input_ids, next_tok], axis=1)
            mx.eval(input_ids)

        return input_ids
