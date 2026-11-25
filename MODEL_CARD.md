---
license: mit
tags:
  - research-prototype
  - pretraining-architecture
  - block-diffusion
  - linear-attention
  - wreath-deltanet
  - self-play
  - pytorch
library_name: pytorch
---

# Philosopher Micro-RSI (`1.52M` Seed Checkpoint & Training Starter Architecture)

**An innovative, hackable, from-scratch PyTorch architecture and `1.52M`-parameter (`5.8 MB`) seed checkpoint built as a great starting point for anyone looking to experiment with or scale up next-generation weight training.**

> **Honest Scope:** This `1.52M`-parameter checkpoint is **not** a finished general-purpose chat model. It is an **open, lightweight architectural seed & training starting point**—small enough to inspect, modify, and train from scratch in minutes on a single consumer GPU or Apple Silicon Mac, while already wiring together cutting-edge sub-quadratic and parallel-diffusion mechanisms so you can immediately scale up width (`d_model`), heads (`n_heads`), and data.

---

## Why Use This as a Starting Point for Model Weight Training?

Most open starter repos hand you a plain vanilla GPT-2 transformer. **Philosopher Micro-RSI** gives researchers, indie builders, and students a clean, single-file PyTorch backbone (`philosopher/brain.py`) that already unifies six modern architectural ideas in one trainable graph:

1. **Dual-Gear `Wreath-KDA` State Tracking (`O(N)` Linear Attention without `BF16`/`FP8` Drift):** Pairs `8×8` `{-1, 0, +1}` signed-permutation **Swap-Boxes** (`L_4` Stiefel valley alignment) with **Unit-Circle `2D` `30°` Clock-Dials**, snapped only at `C`-token chunk boundaries (`gpu_chunk_scan_swap` / `gpu_chunk_scan_clock`) so rounding drift resets to zero (`L* = ∞`). See the companion theory & `682×` extrapolation proofs at [Manageriii/wreath-deltanet](https://github.com/Manageriii/wreath-deltanet).
2. **`5%` Write & Address Dead-Zone Gates:** Allows distractor tokens to execute exact algebraic identities (`S_t = S_{t-1}`) with zero floating-point leakage across long sequences.
3. **Asymmetric Weight-Shared Causal Encoder + `mHC` Manifold Scaling:** Loops 3 shared `RecursiveEncoderKDAGroup` blocks across 19 semantic layers (`6 + 6 + 7`) with learned per-layer depth embeddings and `1 / sqrt(19)` manifold scaling—giving deep effective receptive fields at a fraction of standard parameter cost.
4. **Digit-Invariant `Engram` 2-Gram Memory & Compositional `SuperBPE + CoBPE` Embeddings:** Factors token representations into `(base, modifier)` embeddings plus a lightweight `O(1)` 2-gram lookup bank.
5. **4-Pass Parallel Block-Diffusion Decoder (`NCM + TRM`) with Confidence Locking:** Denoises `40` supertokens (`28` `<think>` + `12` `<ans>`) in parallel across coarse-to-fine passes using `AttnRes` depth residuals, `Stable LatentMoE`, progressive confidence anchor locking, and 4 Top-2 routed task-specialist blocks (`Invention`, `CodeState`, `LogicSAT`, `AITheory`).
6. **Built-In Deterministic Self-Play (`Challenger–Solver`) Loop:** Ships with a zero-leakage sandbox verifier (`philosopher/rsi_engine.py`, `sandbox_verifier.py`) so you can train weights via verifiable self-play right out of the box.

---

## Included Seed Checkpoint & Scoreboard (`1.52M` Params)

Trained from scratch on a single 24 GB Apple M4 Pro laptop across 10 self-play sessions (`623` self-discovered verified solutions on frozen procedural benchmarks):

| Metric (`vault/fixed_exam.json`) | Birth (`S0`) | After 10 Self-Play Sessions (`brain.pt`) |
| :--- | :---: | :---: |
| **Parameters / Disk Size** | `1,521,659` (`5.8 MB`) | `1,521,659` (`5.8 MB`) |
| **Level 1 Exact Accuracy** | `20.0%` | **`40.0%`** *(2.0× gain)* |
| **Level 2 Exact Accuracy** | `10.0%` | **`14.4%`** |
| **Level 3 Exact Accuracy** | `4.4%` | `3.3%` |

---

## Scaling Up for Your Own Weight Training

Clone the full codebase ([https://github.com/Manageriii/philosopher-micro-rsi](https://github.com/Manageriii/philosopher-micro-rsi)) and scale `PhilosopherBrain` cleanly to `50M`, `300M`, or `1B+` parameters by adjusting the constructor arguments in `philosopher/brain.py`:

```python
from philosopher.brain import PhilosopherBrain

# Example: scaling up from the 1.52M laptop seed to a wider pretraining run
model = PhilosopherBrain(
    base_vocab=16384,   # swap in your own BPE / SuperBPE vocab
    mod_vocab=64,
    mask_base_id=3,
    d_model=768,        # scale width (seed uses 128)
    n_heads=12,         # scale parallel query heads
    d_swap=8,           # signed-permutation Swap-Box dimension
    d_v=64,             # per-head DeltaNet value dimension
    n_dials=8,          # 2D modular Clock-Dials
    chunk_c=64,         # C=64 for BF16, C=4 for 8-bit FP8
    block_len=64,       # parallel block-diffusion denoising width
)
```

## Links & Attribution
* **Full Source Code & Interactive Playground (`playground.py`):** https://github.com/Manageriii/philosopher-micro-rsi
* **Companion Proofs & `682×` Extrapolation Suite (`Dual-Gear Wreath-DeltaNet`):** https://github.com/Manageriii/wreath-deltanet
* **License:** MIT (free for academic, personal, and commercial weight-training experiments)
