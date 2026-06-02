# OpenMythos-MLX 🍎🐦‍⬛

**OpenMythos ported to Apple Silicon MLX** — Recurrent-Depth Transformer with MoE, MLA, ACT halting.

[![MLX](https://img.shields.io/badge/MLX-Apple%20Silicon-blue)](https://github.com/ml-explore/mlx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Disclaimer:** OpenMythos is an independent, community-driven theoretical reconstruction. Not affiliated with Anthropic.

## What This Is

A pure MLX implementation of the OpenMythos architecture — a Recurrent-Depth Transformer (RDT) that uses **the same weights looped T times** for deeper reasoning without parameter growth.

**Original:** [github.com/DeadByDawn101/OpenMythos](https://github.com/DeadByDawn101/OpenMythos) (PyTorch)
**This Port:** Pure MLX for Apple Silicon (M1/M2/M3/M4)

## Architecture

```
Input Tokens
     ↓
[Prelude]          — 2 standard transformer blocks (run once)
     ↓
[Recurrent Block]  — 1 transformer block looped T times
     ↑_______↓      h_{t+1} = A·h_t + B·e + Transformer(h_t, e)
     ↓               + ACT halting (adaptive depth per token)
[Coda]             — 2 standard transformer blocks (run once)
     ↓
Output Logits
```

## Key Features

| Feature | Description |
|---------|-------------|
| **MLA** | Multi-Latent Attention (DeepSeek-style compressed KV cache) |
| **GQA** | Grouped Query Attention (fallback) |
| **MoE FFN** | DeepSeek-style shared + routed experts |
| **ACT** | Adaptive Computation Time (variable depth per token) |
| **LTI Injection** | Stable recurrence (spectral radius < 1 guaranteed) |
| **LoRA Depth** | Per-iteration LoRA adaptation |
| **RoPE** | Rotary position embeddings |
| **KV Cache** | Full autoregressive generation support |

## Quick Start

```python
import mlx.core as mx
from open_mythos_mlx import OpenMythos, mythos_1b

cfg = mythos_1b()  # 1B param config
model = OpenMythos(cfg)

ids = mx.random.randint(0, cfg.vocab_size, (1, 16))
logits = model(ids, n_loops=8)
print(f"Logits: {logits.shape}")

# Generate tokens
out = model.generate(ids, max_new_tokens=32, n_loops=16)
```

## Variants

| Config | Params | Dim | Experts | Loop Iters | Context |
|--------|--------|-----|---------|------------|---------|
| `mythos_1b()` | ~1B | 2048 | 64 | 16 | 4K |
| `mythos_3b()` | ~3B | 3072 | 64 | 16 | 4K |
| `mythos_10b()` | ~10B | 4096 | 128 | 24 | 8K |

## Depth Extrapolation

The key property: **train on N loops, test on N+k loops.** More loops = deeper reasoning.

```python
# Train with 8 loops
logits = model(ids, n_loops=8)

# Test with 32 loops — extrapolates to harder problems
logits = model(ids, n_loops=32)
```

## Installation

```bash
pip install -e .
```

## Credits

- Original PyTorch: [OpenMythos](https://github.com/DeadByDawn101/OpenMythos)
- MLX Port: [@DeadByDawn101](https://github.com/DeadByDawn101) / RavenX LLC

## License

MIT
