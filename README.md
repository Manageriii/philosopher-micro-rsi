# Philosopher Micro-RSI (`1.52M`-Parameter Experimental Prototype)

**A `1.52M`-parameter from-scratch PyTorch prototype exploring Dual-Gear `Wreath-KDA`, `40`-token parallel block diffusion (`NCM + TRM`), and verified `Challenger–Solver` self-play on a 24 GB Apple Silicon laptop.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Parameters: 1.52M](https://img.shields.io/badge/Parameters-1.52M%20(5.8%20MB)-orange.svg)](#checkpoint--scoreboard)
[![Status: Research Prototype](https://img.shields.io/badge/Status-Micro--Scale%20Research%20Prototype-blue.svg)](#limitations--honest-scope)

> **Honest Scope:** This is **not** a general-purpose chat LLM. It is a **1.52M-parameter (`5.8 MB`) micro-scale architecture & self-play sandbox** built to test whether a tiny weight-shared Wreath + Block-Diffusion model can discover verified multi-step solutions from scratch via deterministic sandbox feedback on a single laptop. See [wreath-deltanet](https://github.com/Manageriii/wreath-deltanet) for the formal proofs and `682×` extrapolation benchmarks. But for people that have enough computing power I would say this model is a great starting point for training a really strong model because of it's innovative and efficient architecture.

---

## Architecture (`philosopher/brain.py`)

* **Total Parameters:** `1,521,659` (`1.52M`, `checkpoints/brain.pt` = `5.8 MB`)
* **Asymmetric Weight-Shared Causal Encoder:** Looped `KDA` horizon-mixing blocks with per-layer depth embeddings + `mHC` scaling, feeding a Dual-Gear `Wreath-KDA` layer (`8×8` `{-1, 0, +1}` Swap-Boxes + four `30°` 12-hour Clock-Dials + `5%` Dead-Zone Gates + digit-invariant `Engram` 2-gram memory).
* **4-Pass Parallel Block-Diffusion Decoder (`40` supertokens):** Two anchored zones (`28`-token `<think>` zone + `12`-token `<ans>` zone) denoised in parallel across `Pass 0 (NCM Plan) → TRM Pass 1–3`, paired with `AttnRes`, compact `LatentMoE`, and 4 Top-2 routed specialist blocks (`Invention`, `CodeState`, `LogicSAT`, `AITheory`).
* **Deterministic Challenger–Solver Loop (`philosopher/rsi_engine.py`, `sandbox_verifier.py`):** Generates procedural `induce:`, `py:`, and `story:` puzzles, samples `K = 8` parallel diffusion trajectories per puzzle, and admits only deterministically verified solutions into the replay buffer.

---

## Checkpoint & Scoreboard (`10` Self-Play Sessions on M4 Pro)

Measured on the frozen evaluation split (`vault/fixed_exam.json`) across 10 self-play sessions from birth (`S0`) to Session `#10` (`623` self-discovered verified solutions):

| Session | Frontier | Level 1 Exact | Level 2 Exact | Level 3 Exact | `Solved / 240` |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Birth (`S0`)** | Level 1 | `20.0%` | `10.0%` | `4.4%` | *(foundational)* |
| **#3** | Level 1 | `33.3%` | `10.0%` | `1.1%` | `61 / 240` |
| **#7** | Level 1 | `38.9%` | `15.6%` | `2.2%` | `55 / 240` |
| **#9** | Level 1 | **`40.0%`** | `14.4%` | `3.3%` | **`80 / 240`** |
| **#10** | Level 1 | **`40.0%`** | **`14.4%`** | **`3.3%`** | `60 / 240` |

---

## Quick Start

The repository ships with the `5.8 MB` checkpoint (`checkpoints/brain.pt`), tokenizer (`checkpoints/tokenizer.json`), and frozen procedural evaluation suite (`vault/fixed_exam.json`, `vault/rsi_state.json`) so you can inspect the scoreboard and interactive denoiser immediately:

```bash
pip install torch

# 1. Inspect the 10-session scoreboard
python3 run_rsi_session.py --status

# 2. Watch all 40 <MASK> supertokens denoise in parallel (Pass 0 -> Pass 3)
python3 playground.py
# Try inside playground: `challenge`, `challenge 2`, `stats`, or a custom `induce:` / `story:` / `py:` prompt!
```

---

## Limitations & Honest Scope

* **Micro-scale capacity (`1.52M` params):** At 1.52M parameters, the model doubles Level 1 exact accuracy (`20.0% → 40.0%`) but plateaus before unlocking the Level 2/3 frontier (`14.4%` / `3.3%`).
* **Training data note:** External curated corpora are intentionally excluded from this repository; `vault/fixed_exam.json` contains only procedural, self-contained synthetic benchmark puzzles.
* **Hugging Face:** See [MODEL_CARD.md](MODEL_CARD.md) if mirroring `checkpoints/brain.pt` and `checkpoints/tokenizer.json` to Hugging Face Hub.
