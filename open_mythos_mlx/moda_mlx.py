"""
moda_mlx.py — Mixture-of-Depths Attention + DeepSeek MoE for MLX

Port of the MoDA architecture from PyTorch to Apple Silicon MLX.

KEY INNOVATION: Each attention head jointly attends (single softmax) to:
  1. Sequence KVs at the current layer (standard causal GQA)
  2. Depth KVs from ALL preceding layers at the same token position

This means the model can SEE ITS OWN THINKING from previous layers!
Combined with DeepSeek-style MoE (shared + routed experts).

Papers:
  - MoDA: arXiv 2603.15619
  - DeepSeekMoE: arXiv 2401.06066

Original: https://github.com/DeadByDawn101/OpenMythos
MLX Port: RavenX LLC / @DeadByDawn101
"""

import math
from dataclasses import dataclass
from typing import Optional, List, Tuple

import mlx.core as mx
import mlx.nn as nn


@dataclass
class MoDAConfig:
    vocab_size: int = 32_000
    d_model: int = 2048
    n_layers: int = 24
    n_heads_q: int = 16
    n_heads_kv: int = 8
    head_dim: int = 128
    max_seq_len: int = 4_096
    rope_base: float = 10_000.0
    norm_eps: float = 1e-6
    # DeepSeek MoE
    n_shared_experts: int = 2
    n_routed_experts: int = 64
    n_activated_experts: int = 6
    expert_hidden_dim: int = 704
    moe_balance_alpha: float = 0.001
    moe_route_scale: float = 1.0


# ── RoPE ──────────────────────────────────────────────────────────────────

def precompute_rope(dim: int, max_len: int, base: float = 10000.0):
    freqs = 1.0 / (base ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
    t = mx.arange(max_len, dtype=mx.float32)
    angles = mx.outer(t, freqs)
    return mx.cos(angles), mx.sin(angles)


def apply_rope(x, cos_f, sin_f):
    B, H, T, D = x.shape
    half = D // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos_f = cos_f[:T][None, None, :, :]
    sin_f = sin_f[:T][None, None, :, :]
    return mx.concatenate([x1 * cos_f - x2 * sin_f, x1 * sin_f + x2 * cos_f], axis=-1)


# ── DeepSeek Expert ───────────────────────────────────────────────────────

class DeepSeekExpert(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)

    def __call__(self, x):
        return self.w2(nn.silu(self.w1(x)) * self.w3(x))


# ── DeepSeek Gate ─────────────────────────────────────────────────────────

class DeepSeekGate(nn.Module):
    def __init__(self, d_model: int, n_experts: int, n_activated: int, route_scale: float = 1.0):
        super().__init__()
        self.n_experts = n_experts
        self.n_activated = n_activated
        self.route_scale = route_scale
        self.weight = mx.random.normal((n_experts, d_model)) * 0.02

    def __call__(self, x):
        logits = x @ self.weight.T
        scores = mx.softmax(logits, axis=-1)
        indices = mx.argpartition(-scores, kth=self.n_activated, axis=-1)[..., :self.n_activated]
        weights = mx.take_along_axis(scores, indices, axis=-1)
        weights = weights * self.route_scale
        return weights, indices, scores


# ── DeepSeek MoE ──────────────────────────────────────────────────────────

class DeepSeekMoE(nn.Module):
    def __init__(self, cfg: MoDAConfig):
        super().__init__()
        shared_hidden = cfg.n_shared_experts * cfg.expert_hidden_dim
        self.shared = DeepSeekExpert(cfg.d_model, shared_hidden)
        self.gate = DeepSeekGate(cfg.d_model, cfg.n_routed_experts, cfg.n_activated_experts, cfg.moe_route_scale)
        self.experts = [DeepSeekExpert(cfg.d_model, cfg.expert_hidden_dim) for _ in range(cfg.n_routed_experts)]
        self.n_activated = cfg.n_activated_experts
        self.balance_alpha = cfg.moe_balance_alpha

    def __call__(self, x):
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)

        # Shared experts (always active)
        shared_out = self.shared(x_flat)

        # Routed experts (sparse)
        weights, indices, scores = self.gate(x_flat)
        routed_out = mx.zeros_like(x_flat)

        for k in range(self.n_activated):
            idx = indices[:, k]
            w = weights[:, k:k+1]
            for e_idx in range(len(self.experts)):
                mask = (idx == e_idx)
                if mx.any(mask):
                    expert_in = x_flat * mask[:, None].astype(x_flat.dtype)
                    routed_out = routed_out + w * self.experts[e_idx](expert_in) * mask[:, None].astype(x_flat.dtype)

        return (shared_out + routed_out).reshape(B, T, D)


# ── MoDA Attention (THE KEY INNOVATION) ───────────────────────────────────

class MoDAAttention(nn.Module):
    """Mixture-of-Depths Attention.

    Each query jointly attends (single softmax) to:
      * Sequence KVs at current layer (causal GQA)
      * Depth KVs from ALL preceding layers at same token position

    The model can SEE ITS OWN THINKING from previous layers!
    """

    def __init__(self, cfg: MoDAConfig):
        super().__init__()
        self.n_heads_q = cfg.n_heads_q
        self.n_heads_kv = cfg.n_heads_kv
        self.head_dim = cfg.head_dim
        self.gqa_group = cfg.n_heads_q // cfg.n_heads_kv
        self.scale = cfg.head_dim ** -0.5

        inner_q = cfg.n_heads_q * cfg.head_dim
        inner_kv = cfg.n_heads_kv * cfg.head_dim

        self.q_proj = nn.Linear(cfg.d_model, inner_q, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, inner_kv, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, inner_kv, bias=False)
        self.o_proj = nn.Linear(inner_q, cfg.d_model, bias=False)

    def _expand_kv(self, kv):
        if self.gqa_group == 1:
            return kv
        return mx.repeat(kv, self.gqa_group, axis=1)

    def __call__(self, x, depth_k_cache, depth_v_cache, cos_f, sin_f):
        B, T, D = x.shape
        Hq, Hk, d = self.n_heads_q, self.n_heads_kv, self.head_dim

        Q = self.q_proj(x).reshape(B, T, Hq, d).transpose(0, 2, 1, 3)
        K = self.k_proj(x).reshape(B, T, Hk, d).transpose(0, 2, 1, 3)
        V = self.v_proj(x).reshape(B, T, Hk, d).transpose(0, 2, 1, 3)

        Q = apply_rope(Q, cos_f, sin_f)
        K = apply_rope(K, cos_f, sin_f)

        K_e = self._expand_kv(K)
        V_e = self._expand_kv(V)

        L = len(depth_k_cache)

        if L == 0:
            # Standard causal attention (no depth cache yet)
            scores = (Q @ K_e.transpose(0, 1, 3, 2)) * self.scale
            mask = mx.full((1, 1, T, T), float('-inf'))
            mask = mx.triu(mask, k=1)
            scores = scores + mask
            attn = mx.softmax(scores, axis=-1)
            out = attn @ V_e
        else:
            # === JOINT SEQUENCE + DEPTH ATTENTION ===
            # Sequence logits [B, Hq, T, T] with causal mask
            seq_logits = (Q @ K_e.transpose(0, 1, 3, 2)) * self.scale
            causal_mask = mx.full((1, 1, T, T), float('-inf'))
            causal_mask = mx.triu(causal_mask, k=1)
            seq_logits = seq_logits + causal_mask

            # Depth KVs: stack from all preceding layers
            # Each entry: [B, Hk, T, d] → stack to [B, Hk, L, T, d] → permute to [B, Hk, T, L, d]
            K_depth = mx.stack(depth_k_cache, axis=2).transpose(0, 1, 3, 2, 4)
            V_depth = mx.stack(depth_v_cache, axis=2).transpose(0, 1, 3, 2, 4)
            K_depth_e = self._expand_kv(K_depth)
            V_depth_e = self._expand_kv(V_depth)

            # Depth logits [B, Hq, T, L]
            # Q: [B, Hq, T, d], K_depth_e: [B, Hq, T, L, d]
            depth_logits = mx.sum(Q[:, :, :, None, :] * K_depth_e, axis=-1) * self.scale

            # UNIFIED SOFTMAX over T + L positions (THE KEY!)
            combined = mx.concatenate([seq_logits, depth_logits], axis=-1)
            weights = mx.softmax(combined, axis=-1)

            seq_weights = weights[:, :, :, :T]
            depth_weights = weights[:, :, :, T:]

            seq_contrib = seq_weights @ V_e
            # depth_contrib: [B, Hq, T, L] × [B, Hq, T, L, d] → [B, Hq, T, d]
            depth_contrib = mx.sum(depth_weights[:, :, :, :, None] * V_depth_e, axis=3)

            out = seq_contrib + depth_contrib

        out = out.transpose(0, 2, 1, 3).reshape(B, T, Hq * d)
        return self.o_proj(out)


# ── MoDA Block ────────────────────────────────────────────────────────────

class MoDABlock(nn.Module):
    """Single MoDA + DeepSeek-MoE transformer block.

    After processing, writes depth KV cache entries for the next layer.
    """

    def __init__(self, cfg: MoDAConfig):
        super().__init__()
        inner_kv = cfg.n_heads_kv * cfg.head_dim

        self.attn = MoDAAttention(cfg)
        self.moe = DeepSeekMoE(cfg)
        self.norm_attn = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.norm_ffn = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)

        # Depth cache write projections
        self.k_write = nn.Linear(cfg.d_model, inner_kv, bias=False)
        self.v_write = nn.Linear(cfg.d_model, inner_kv, bias=False)

        self._n_heads_kv = cfg.n_heads_kv
        self._head_dim = cfg.head_dim

    def __call__(self, x, depth_k_cache, depth_v_cache, cos_f, sin_f):
        B, T, _ = x.shape

        # Post-norm attention
        x = self.norm_attn(x + self.attn(x, depth_k_cache, depth_v_cache, cos_f, sin_f))

        # Post-norm MoE
        x = self.norm_ffn(x + self.moe(x))

        # Depth write: project block output to depth KV
        k_w = self.k_write(x).reshape(B, T, self._n_heads_kv, self._head_dim).transpose(0, 2, 1, 3)
        v_w = self.v_write(x).reshape(B, T, self._n_heads_kv, self._head_dim).transpose(0, 2, 1, 3)

        # Apply RoPE to depth keys for positional consistency
        k_w = apply_rope(k_w, cos_f, sin_f)

        return x, k_w, v_w


# ── Full MoDA Model ──────────────────────────────────────────────────────

class MoDAModel(nn.Module):
    """Full MoDA + DeepSeek-MoE language model.

    Each layer attends to BOTH:
      - Current sequence (causal, like standard transformer)
      - ALL preceding layers' outputs (depth attention)

    This gives the model access to its own intermediate representations,
    enabling deeper reasoning without additional parameters.
    """

    def __init__(self, cfg: MoDAConfig):
        super().__init__()
        self.cfg = cfg

        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.cos_freqs, self.sin_freqs = precompute_rope(cfg.head_dim, cfg.max_seq_len, cfg.rope_base)
        self.blocks = [MoDABlock(cfg) for _ in range(cfg.n_layers)]
        self.norm_out = nn.RMSNorm(cfg.d_model, eps=cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

    def __call__(self, input_ids):
        B, T = input_ids.shape

        x = self.embed(input_ids)
        cos_f = self.cos_freqs[:T]
        sin_f = self.sin_freqs[:T]

        # Depth KV cache — grows with each layer
        depth_k_cache = []
        depth_v_cache = []

        for block in self.blocks:
            x, k_write, v_write = block(x, depth_k_cache, depth_v_cache, cos_f, sin_f)
            depth_k_cache.append(k_write)
            depth_v_cache.append(v_write)

        x = self.norm_out(x)
        return self.lm_head(x)


# ── Config Presets ────────────────────────────────────────────────────────

def moda_small() -> MoDAConfig:
    """~400M MoDA model for testing."""
    return MoDAConfig(
        d_model=1024, n_layers=12, n_heads_q=8, n_heads_kv=4,
        head_dim=128, n_routed_experts=32, n_activated_experts=4,
        expert_hidden_dim=512,
    )

def moda_2b() -> MoDAConfig:
    """~2B MoDA model — equivalent to the paper's recommended config."""
    return MoDAConfig(
        d_model=2048, n_layers=24, n_heads_q=16, n_heads_kv=8,
        head_dim=128, n_routed_experts=64, n_activated_experts=6,
        expert_hidden_dim=704,
    )

def moda_7b() -> MoDAConfig:
    """~7B MoDA model — production scale."""
    return MoDAConfig(
        d_model=4096, n_layers=32, n_heads_q=32, n_heads_kv=8,
        head_dim=128, n_routed_experts=128, n_activated_experts=8,
        expert_hidden_dim=1024,
    )
