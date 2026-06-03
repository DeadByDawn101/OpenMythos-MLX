"""
finetune_security.py — Fine-tune maidacundo's pretrained OpenMythos 140M
on RavenX security data + Mythos character distillation.

Then generate deep traces at 8x-32x depth for distillation into 35B.

Usage:
  HF_TOKEN=hf_... python3.13 finetune_security.py

Author: RavenX LLC / @DeadByDawn101
"""
import sys, os, json, time
import torch
from huggingface_hub import snapshot_download

# ── Step 1: Load pretrained model ──
print("="*60)
print("  OPENMYTHOS SECURITY FINE-TUNING + DISTILLATION")
print("="*60)

HF_TOKEN = os.environ.get("HF_TOKEN", None)
code_path = snapshot_download("maidacundo/open-mythos-hf", token=HF_TOKEN)
model_path = snapshot_download("maidacundo/open-mythos-140m", token=HF_TOKEN)
sys.path.insert(0, code_path)

# Fix config
config_file = os.path.join(model_path, "config.json")
with open(config_file) as f:
    cfg = json.load(f)
cfg.pop("tie_word_embeddings", None)
with open(config_file, "w") as f:
    json.dump(cfg, f, indent=2)

from open_mythos_hf import OpenMythosForCausalLM, OpenMythosConfig
from transformers import AutoTokenizer

config = OpenMythosConfig.from_pretrained(model_path)
model = OpenMythosForCausalLM.from_pretrained(model_path, config=config)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Ensure pad token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

total = sum(p.numel() for p in model.parameters())
print(f"\nLoaded: {total:,} params")
print(f"Recurrence: {config.mean_recurrence}")
print(f"Architecture: prelude={config.n_layers_in_prelude}, "
      f"recurrent={config.n_layers_in_recurrent_block}, "
      f"coda={config.n_layers_in_coda}")

# ── Step 2: Load security training data ──
print(f"\nStep 2: Loading security data...")
data_path = os.path.expanduser("~/Developer/RavenX-Sec/data/train.jsonl")
mythos_path = os.path.expanduser("~/Developer/RavenX-Sec/data/extracted/mythos_character_distill_551.jsonl")

texts = []

# Load main security data (first 2000 for quick test)
if os.path.exists(data_path):
    with open(data_path) as f:
        for i, line in enumerate(f):
            if i >= 2000: break
            try:
                item = json.loads(line)
                msgs = item.get("messages", [])
                text = " ".join(m.get("content", "") for m in msgs)
                if len(text) > 100:
                    texts.append(text[:512])
            except: pass
    print(f"  Security data: {len(texts)} examples")

# Load Mythos character distillation (ALL 551)
mythos_count = 0
if os.path.exists(mythos_path):
    with open(mythos_path) as f:
        for line in f:
            try:
                item = json.loads(line)
                msgs = item.get("messages", [])
                text = " ".join(m.get("content", "") for m in msgs)
                if len(text) > 50:
                    texts.append(text[:512])
                    mythos_count += 1
            except: pass
    print(f"  Mythos character: {mythos_count} examples")
else:
    # Try downloading from HF
    try:
        from huggingface_hub import hf_hub_download
        dl = hf_hub_download(
            "deadbydawn101/ravenx-sec-training-data",
            "mythos_character_distill_551.jsonl",
            repo_type="dataset", token=HF_TOKEN
        )
        with open(dl) as f:
            for line in f:
                try:
                    item = json.loads(line)
                    msgs = item.get("messages", [])
                    text = " ".join(m.get("content", "") for m in msgs)
                    if len(text) > 50:
                        texts.append(text[:512])
                        mythos_count += 1
                except: pass
        print(f"  Mythos character (from HF): {mythos_count} examples")
    except: pass

print(f"  Total: {len(texts)} training examples")

# ── Step 3: Fine-tune ──
print(f"\nStep 3: Fine-tuning on security data...")

# Use MPS (Apple Silicon) if available
device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)
print(f"  Device: {device}")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

# Warmup schedule
WARMUP = 20
TOTAL = 200
PEAK_LR = 5e-5

import math
def get_lr(step):
    if step < WARMUP:
        return PEAK_LR * (step + 1) / WARMUP
    progress = (step - WARMUP) / max(1, TOTAL - WARMUP)
    return PEAK_LR * 0.5 * (1 + math.cos(math.pi * progress))

t0 = time.time()
losses = []

for step in range(TOTAL):
    lr = get_lr(step)
    for pg in optimizer.param_groups:
        pg["lr"] = lr

    text = texts[step % len(texts)]
    tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    tokens = {k: v.to(device) for k, v in tokens.items()}

    output = model(input_ids=tokens["input_ids"], labels=tokens["input_ids"])
    loss = output.loss

    if torch.isnan(loss):
        print(f"  Step {step}: NaN! lr={lr:.2e}")
        break

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()

    losses.append(loss.item())
    if step % 20 == 0:
        print(f"  Step {step:4d}: loss={loss.item():.4f} lr={lr:.2e} ({time.time()-t0:.1f}s)")

dt = time.time() - t0
print(f"  Training done in {dt:.1f}s")
print(f"  Final loss: {losses[-1]:.4f}" if losses else "  No training completed")

# ── Step 4: Generate deep traces at different depths ──
if losses and not torch.isnan(torch.tensor(losses[-1])):
    print(f"\nStep 4: Generating deep traces at different depths...")
    model.eval()

    security_prompts = [
        "Security assessment of an open MongoDB 4.2 instance on port 27017 with no authentication containing customer PII",
        "Vulnerability analysis of Kubernetes API server with anonymous authentication and privileged pods",
        "Security review of Jenkins CI/CD pipeline storing AWS credentials in plaintext environment variables",
        "Penetration test findings for PostgreSQL 12.3 database exposed on port 5432 with default credentials",
        "Bug bounty report for AWS S3 bucket with public read access and IAM role with AdministratorAccess",
    ]

    traces = []
    for prompt in security_prompts:
        print(f"\n  Prompt: {prompt[:60]}...")
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        for depth_label in ["default"]:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )
            text_out = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"    Depth={depth_label}: {text_out[:150]}...")

            traces.append({
                "messages": [
                    {"role": "system", "content": "You are RavenX-Sec with OpenMythos deep reasoning. Follow 6-step RATH."},
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": text_out}
                ]
            })

    # Save traces
    output_file = os.path.expanduser("~/Developer/RavenX-Sec/data/extracted/openmythos_deep_traces.jsonl")
    with open(output_file, "w") as f:
        for t in traces:
            f.write(json.dumps(t) + "\n")
    print(f"\nSaved {len(traces)} deep traces → {output_file}")

    # ── Step 5: Save fine-tuned model ──
    save_path = os.path.expanduser("~/Developer/OpenMythos-MLX/models/openmythos-140m-security")
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved → {save_path}")

print(f"\n{'='*60}")
print(f"  PIPELINE COMPLETE")
print(f"  Pretrained: maidacundo/open-mythos-140m")
print(f"  Fine-tuned on: {len(texts)} security + Mythos examples")
print(f"  Traces saved: ready for 35B distillation")
print(f"{'='*60}")
