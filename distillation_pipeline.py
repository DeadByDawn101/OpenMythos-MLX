"""
RavenX OpenMythos Distillation Pipeline
========================================
1. Load maidacundo's pretrained 140M OpenMythos
2. Fine-tune on RavenX security data + Mythos character distillation
3. Generate deep traces at 8x depth
4. Save as training data for 35B distillation

Author: RavenX LLC / @DeadByDawn101
"""
import json, os, time

print("="*60)
print("  RAVENX OPENMYTHOS DISTILLATION PIPELINE")
print("="*60)

# Step 1: Download maidacundo's pretrained model
print("\nStep 1: Downloading maidacundo/open-mythos-140m...")
from huggingface_hub import snapshot_download
model_path = snapshot_download(
    "maidacundo/open-mythos-140m",
    token=os.environ.get("HF_TOKEN", None)
)
print(f"  Downloaded to: {model_path}")

# Check what files we got
for f in sorted(os.listdir(model_path)):
    fpath = os.path.join(model_path, f)
    if os.path.isfile(fpath):
        size = os.path.getsize(fpath)
        print(f"  {f}: {size/1024/1024:.1f} MB")

# Step 2: Load the config
print("\nStep 2: Inspecting model architecture...")
config_path = os.path.join(model_path, "config.json")
if os.path.exists(config_path):
    with open(config_path) as f:
        config = json.load(f)
    for k, v in config.items():
        if not isinstance(v, (dict, list)):
            print(f"  {k}: {v}")
        elif isinstance(v, dict) and len(v) < 10:
            print(f"  {k}: {v}")

print("\nStep 1-2 COMPLETE!")
print("Paste output so we can build the fine-tuning step!")
