# Depth-Extrapolated Reasoning for Domain-Specific AI Models

**Author:** Gabriel Garcia, RavenX LLC
**Date:** June 2026
**Status:** Research Summary (detailed methodology in private repository)

---

## Abstract

We demonstrate that Recurrent-Depth Transformers (RDTs) trained on domain-specific data exhibit depth extrapolation on consumer Apple Silicon hardware — training at depth T and achieving improved performance at depth 4T. We further show that reasoning patterns learned at extrapolated depth can be transferred to standard production architectures (MoE, dense transformers) via trace distillation, enabling deeper reasoning without inference-time architectural changes. We validate this approach on cybersecurity assessment tasks, producing models that generate multi-hop security analyses in a single forward pass.

## Key Results

### 1. Depth Extrapolation on Apple Silicon

First demonstration of RDT depth extrapolation on consumer hardware (Apple M4 Max, 128GB, MLX framework):

| Training Depth | Test Depth | Loss | Result |
|---------------|-----------|------|--------|
| T=2 | T=1 | 10.3155 | Baseline |
| T=2 | T=2 | 10.2770 | Training depth |
| T=2 | T=4 | 10.2448 | Better (extrapolated) |
| T=2 | T=8 | 10.2380 | Best (4x extrapolation) |
| T=2 | T=16 | 10.2488 | Slight degradation |

Training at depth 2, optimal performance at depth 8 — a 4x depth extrapolation ratio.

### 2. Reasoning Transfer via Distillation

Reasoning traces generated at extrapolated depth (8x) were distilled into a standard 35B Mixture-of-Experts model. The resulting model produces multi-step security assessment reasoning in a single forward pass, with no architectural modifications at inference time.

Production results:
- Model generates structured 6-phase security assessments
- Outputs include real CVE references, CVSS scoring, and remediation code
- Inference speed: 85-91 tokens/second on Apple M4 Max (Q4_K_M quantization)
- Reasoning patterns survive aggressive quantization (BF16 → 4.88 bits per weight)

### 3. Domain Validation: Cybersecurity

The approach was validated on cybersecurity assessment tasks:

| Benchmark Category | Score |
|-------------------|-------|
| Identity | 93.8% |
| Code Generation | 97.9% |
| Reasoning | 85.4% |
| Security Protocol | 70.8% |
| Self-Improvement | 68.8% |
| Overall | 80.9% |

The model produces complete penetration testing reports from single prompts, including reconnaissance findings, exploit chains with proof-of-concept code, business impact quantification, remediation with working code, and prevention architecture recommendations.

### 4. Identity Persistence Through Training

A related finding: structured identity training embeds behavioral characteristics into model weights that persist without system prompts and survive quantization. This suggests that targeted fine-tuning modifies weights at a level deeper than surface pattern matching.

## Architecture Overview

```
Training Phase:
  Domain Data → Small RDT (140M-1B) → Train at T=2
                                      → Generate traces at T=8 (extrapolated)
                                      → Distill into production model

Inference Phase:
  User Query → Standard MoE Model (35B) → Deep reasoning output
               (no RDT architecture needed)
               (standard serving: Ollama, llama.cpp, vLLM)
```

## Reproducibility

- Hardware: Apple M4 Max 128GB (training + inference)
- Framework: MLX (Apple Silicon optimized)
- Base RDT: OpenMythos architecture (kyegomez, MIT license)
- Production model: Qwen3.6-35B-A3B MoE
- All depth extrapolation results are reproducible with the open-source code in this repository

## Relationship to Industry Research

Our work was conducted independently. Subsequent public disclosures by major AI laboratories have validated several parallel concepts, including iterative post-training refinement, chain-of-thought controllability, and the importance of cybersecurity as a top capability domain. Timestamped commits in this repository document our independent development timeline.

## Limitations

- Depth extrapolation validated on a single model size (140M parameters)
- Distillation quality depends on trace diversity and domain coverage
- Quantization persistence needs validation across more architectures
- Security assessment quality should be evaluated by domain experts, not solely by automated benchmarks

## Open Source Components

- RDT architecture implementation (MLX): this repository
- Depth extrapolation validation code: `examples/depth_extrapolation_test.py`
- MoDA (Mixture-of-Depths Attention) port: `open_mythos_mlx/moda_mlx.py`

Detailed training methodology, distillation pipeline specifics, and production training recipes are maintained in a private repository under patent review (USPTO #64/087,357).

---

*Gabriel Garcia / RavenX LLC*
*Independent research. Not affiliated with Anthropic, OpenAI, or Google.*
