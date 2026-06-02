"""Example: Create and run OpenMythos-MLX."""
import mlx.core as mx
from open_mythos_mlx import OpenMythos, MythosConfig, mythos_1b

# Use 1B config
cfg = mythos_1b()
model = OpenMythos(cfg)

total = sum(p.size for p in model.parameters().values())
print(f"Parameters: {total:,}")

# Forward pass
ids = mx.random.randint(0, cfg.vocab_size, (2, 16))
logits = model(ids, n_loops=4)
print(f"Logits shape: {logits.shape}")

# Generate
out = model.generate(ids, max_new_tokens=8, n_loops=8)
print(f"Generated shape: {out.shape}")
