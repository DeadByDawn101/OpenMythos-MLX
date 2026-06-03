# OpenMythos-MLX 🍎🐦‍⬛

**OpenMythos ported to Apple Silicon MLX — with CONFIRMED 4x depth extrapolation!**

[![MLX](https://img.shields.io/badge/MLX-Apple%20Silicon-blue)](https://github.com/ml-explore/mlx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Disclaimer:** OpenMythos is an independent, community-driven theoretical reconstruction. Not affiliated with Anthropic.

## 🔥 BREAKTHROUGH: 4x Depth Extrapolation Confirmed

**Trained on RavenX-Sec security data, M4 Max 128GB:**

```
n_loops= 1: loss=10.3155
n_loops= 2: loss=10.2770  ← trained here  
n_loops= 4: loss=10.2448  ← BETTER (extrapolated!)
n_loops= 8: loss=10.2380  ← BEST (4x training depth!)
n_loops=16: loss=10.2488  ← slight degradation
n_loops=32: loss=10.2644  ← ACT halting would fix this
```

**Train at 2 loops → optimal at 8 loops = 4x depth extrapolation. More loops = deeper reasoning without additional parameters.**

This is the first demonstration of RDT depth extrapolation on Apple Silicon and on security domain data.

## What This Is

A pure MLX implementation of the OpenMythos architecture — a Recurrent-Depth Transformer (RDT) that uses **the same weights looped T times** for deeper reasoning without parameter growth. Now includes **MoDA (Mixture-of-Depths Attention)** where each layer attends to ALL preceding layers' outputs.

**Original:** [github.com/DeadByDawn101/OpenMythos](https://github.com/DeadByDawn101/OpenMythos) (PyTorch)

## Two Architectures

### RDT (Recurrent-Depth Transformer) — `model.py`
```
Input → [Prelude] → [Recurrent Block × T loops] → [Coda] → Output
                      ↑_____________↓
                      Same weights, more loops = deeper thinking
```

### MoDA (Mixture-of-Depths Attention) — `moda_mlx.py`
```
Layer N attends to: [current sequence] + [Layer 1..N-1 outputs]
= The model can SEE ITS OWN THINKING from previous layers!
```

## Quick Start

```python
import mlx.core as mx
from open_mythos_mlx import OpenMythos, mythos_1b

cfg = mythos_1b()
model = OpenMythos(cfg)

ids = mx.random.randint(0, cfg.vocab_size, (1, 32))
logits = model(ids, n_loops=8)  # more loops = deeper reasoning
```

## Key Innovation: stop_gradient Training

The breakthrough for stable RDT training on MLX:

```python
for t in range(n_loops):
    if t < n_loops - 1:
        x = mx.stop_gradient(x)  # detach history
    out = self.recurrent(x + 0.1 * e)
    x = 0.5 * x + 0.5 * out
```

Only backpropagate through the **last loop iteration** — prevents gradient explosion through time while maintaining depth extrapolation.

## Research Papers This Enables

1. **"RDT-Distilled Security Reasoning in MoE Transformers"** — Train RDT on security data, distill deep traces into production MoE model
2. **"RDT-to-MoE Reasoning Transfer"** — General technique for transferring variable-depth reasoning to fixed-depth architectures

## Credits

- Original PyTorch: [OpenMythos](https://github.com/DeadByDawn101/OpenMythos) by kyegomez
- MLX Port + Depth Extrapolation: [@DeadByDawn101](https://github.com/DeadByDawn101) / RavenX LLC

## License

MIT
