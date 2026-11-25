---
license: mit
tags:
  - research-prototype
  - block-diffusion
  - linear-attention
  - wreath-deltanet
  - self-play
  - pytorch
library_name: pytorch
---

# Philosopher Micro-RSI (`1.52M` Checkpoint)

> **Warning / Intended Use:** This is a **`1.52M`-parameter (`5.8 MB`) experimental research checkpoint**, **not** a general-purpose assistant or chat model. It is published strictly for reproducibility and inspection of micro-scale Dual-Gear Wreath-KDA + parallel block-diffusion self-play dynamics.

## Model Details
* **Architecture:** Weight-shared causal KDA encoder + Dual-Gear `Wreath-KDA` (`8×8` signed-permutation Swap-Boxes + four `30°` Clock-Dials) + 4-pass `NCM + TRM` parallel block-diffusion decoder (`40` supertokens).
* **Parameters:** `1,521,659` (`1.52M`)
* **Files:** `brain.pt` (`5.8 MB` PyTorch state dict) and `tokenizer.json` (`SuperBPE + CoBPE` vocabulary).
* **Source Code:** [https://github.com/Manageriii/philosopher-micro-rsi](https://github.com/Manageriii/philosopher-micro-rsi)
* **Companion Theory Paper:** [https://github.com/Manageriii/wreath-deltanet](https://github.com/Manageriii/wreath-deltanet)

## Frozen Evaluation Scoreboard (10 Self-Play Sessions)
* **Level 1 Exact Accuracy:** `20.0%` (Birth) → **`40.0%`** (Session #10)
* **Level 2 Exact Accuracy:** `10.0%` → **`14.4%`**
* **Level 3 Exact Accuracy:** `4.4%` → **`3.3%`**

## Limitations
* Restricted to compact procedural reasoning prompts (`induce:`, `story:`, `py:`) and research inspection via `playground.py`.
* Not suitable for general text generation, production deployment, or factual QA.
