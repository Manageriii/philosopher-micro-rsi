#!/usr/bin/env python3
"""
Philosopher 1.0 — Reasoning-First Corpus Curator
================================================
Scans raw `data/` folders (`rpr_v4` .. `rpr_v12`), filters out:
  - Cross-version duplicates (v12 prioritized down to v4)
  - Multimodal image-dependent rows (`images != []`) & PNG chart folders
  - Synthetic random-UUID KV needle haystacks (`256k_needles`, `long_kv`)
  - Rote JSON-schema / `<tool_call>` API-wrapper boilerplate & no-`<think>` rows
  - Overly rambling multi-page transcripts

And distills every clean episode into a Reasoning-First `(prompt, think, ans)` format
inside `data/philosopher_curated/` where 80%+ of target density is `<think>` derivation
and `<ans>` is stripped down to its compact, verifiable core!
"""

import glob
import json
import os
import re
from collections import Counter


SKIP_FILES = {
    "heldout_long_256k_needles_12.jsonl",
    "train_v11_long_kv.jsonl",
    "train_v12_long_kv.jsonl",
    "long_context_eval_16.jsonl",
    "ai_reasoning_think.jsonl",
    "verifiable_frontier.jsonl",
}

SKIP_PILLARS = {
    "vision_multimodal",
    "schema_json",
    "schema_lines",
    "multistep_tool_use",
    "tool_positive",
    "tool_pos",
}


def rank_file(path: str) -> tuple[int, int, str]:
    for idx, tag in enumerate(["v12", "v11", "v10", "v9", "v8", "v7", "v6", "v5", "v4"]):
        if tag in path:
            tier = 0 if any(k in path for k in ("bespoke", "frontier", "train", "dapo")) else 1
            return (idx, tier, path)
    return (99, 1, path)


def distill_compact_conclusion(raw_ans: str, max_chars: int = 120) -> str:
    """
    Strips giant multi-page markdown essay formatting after `</think>` down to the
    crisp final formula, code signature, or 1-line conclusion so model capacity
    goes into `<think>` reasoning & invention rather than essay verbosity!
    """
    txt = raw_ans.strip()
    # If JSON object at end of answer, grab it
    m_json = re.search(r"(\{[^{}]{4,140}\})\s*$", txt)
    if m_json:
        return re.sub(r"\s+", " ", m_json.group(1)).strip()[:max_chars]

    # If fenced Python function, keep compact signature + key return/core line
    if "```python" in txt:
        code = txt.split("```python", 1)[1].split("```", 1)[0].strip()
        lines = [ln.strip() for ln in code.splitlines() if ln.strip() and not ln.strip().startswith("#")]
        sig = lines[0] if lines else "def fn():"
        ret = next((ln for ln in reversed(lines) if ln.startswith("return ")), lines[-1] if lines else "")
        compact = f"{sig} ... {ret}" if ret != sig else sig
        return compact[:max_chars]

    # Strip markdown headers and whitespace, take concise concluding / summary line
    clean_lines = [
        re.sub(r"^#+\s*", "", ln).strip()
        for ln in txt.splitlines()
        if ln.strip() and not ln.strip().startswith("```")
    ]
    if not clean_lines:
        return "verified"
    # Prefer short concluding verdict line if present, else compact first line
    for ln in reversed(clean_lines):
        if 6 <= len(ln) <= max_chars:
            return re.sub(r"\s+", " ", ln)
    return re.sub(r"\s+", " ", clean_lines[0])[:max_chars]


def distill_think_trace(raw_think: str, max_chars: int = 380) -> str:
    """Normalizes `<think>` whitespace while preserving `[SELF-DISCOVER]` / `[L1..L3]` structure."""
    th = re.sub(r"\s+", " ", raw_think).strip()
    if len(th) <= max_chars:
        return th
    # Cut cleanly at sentence boundary near max_chars
    cut = th[:max_chars]
    last_dot = max(cut.rfind(". "), cut.rfind("; "))
    if last_dot >= int(0.6 * max_chars):
        return cut[: last_dot + 1].strip()
    return cut.strip()


def curate_all(data_root: str) -> dict:
    out_dir = os.path.join(data_root, "philosopher_curated")
    os.makedirs(out_dir, exist_ok=True)

    all_jsonl = [
        p
        for p in glob.glob(os.path.join(data_root, "**", "*.jsonl"), recursive=True)
        if "philosopher_curated" not in p
    ]
    all_jsonl.sort(key=rank_file)

    seen_prompts: set[str] = set()
    dropped = Counter()
    verifiable_rows: list[dict] = []
    reasoning_rows: list[dict] = []
    pillar_counts = Counter()

    for path in all_jsonl:
        bn = os.path.basename(path)
        if bn in SKIP_FILES:
            with open(path, "r", encoding="utf-8") as f:
                dropped["synthetic_kv_needle_haystacks"] += sum(1 for ln in f if ln.strip())
            continue

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                if obj.get("images"):
                    dropped["multimodal_image_rows"] += 1
                    continue
                pillar = obj.get("pillar") or obj.get("kind") or "general_reasoning"
                if pillar in SKIP_PILLARS:
                    dropped["rote_json_schema_or_tool_wrappers"] += 1
                    continue

                msgs = obj.get("messages", [])
                u_txt = "\n".join(str(m.get("content", "")) for m in msgs if m.get("role") == "user").strip()
                a_txt = "\n".join(str(m.get("content", "")) for m in msgs if m.get("role") == "assistant").strip()
                if not a_txt and obj.get("reference_answer"):
                    a_txt = str(obj["reference_answer"]).strip()
                if not a_txt and obj.get("chosen"):
                    a_txt = str(obj["chosen"]).strip()

                if not u_txt or not a_txt:
                    dropped["missing_prompt_or_trace"] += 1
                    continue
                if "<tool_call>" in a_txt or '"tool_calls"' in line:
                    dropped["tool_call_api_boilerplate"] += 1
                    continue

                u_norm = re.sub(r"\s+", " ", u_txt).strip()
                ukey = u_norm[:280]
                if ukey in seen_prompts:
                    dropped["cross_version_duplicates"] += 1
                    continue

                m = re.search(r"<think>(.*?)</think>(.*)", a_txt, flags=re.S)
                if not m:
                    dropped["answer_only_without_think"] += 1
                    continue

                raw_th, raw_an = m.group(1).strip(), m.group(2).strip()
                if len(raw_th) < 40:
                    dropped["trivial_short_think"] += 1
                    continue
                if len(u_norm) > 1600 or len(raw_th) > 1400:
                    dropped["overlong_rambling_transcripts"] += 1
                    continue

                seen_prompts.add(ukey)
                th_clean = distill_think_trace(raw_th, max_chars=380)
                p_short = u_norm[:520]

                ver = obj.get("verifier") or {}
                exp = ver.get("expected") if isinstance(ver, dict) else None
                if exp is None and isinstance(obj.get("expected"), dict):
                    exp = obj.get("expected")

                pillar_counts[pillar] += 1
                if exp is not None:
                    ans_compact = json.dumps(exp, separators=(",", ":"), sort_keys=True)
                    if len(ans_compact) <= 120:
                        verifiable_rows.append(
                            {
                                "domain": "ai_verifiable",
                                "pillar": pillar,
                                "level": 2,
                                "prompt": f"reason: {p_short} ->",
                                "think": th_clean,
                                "ans": ans_compact,
                            }
                        )
                        continue

                ans_distilled = distill_compact_conclusion(raw_an, max_chars=110)
                reasoning_rows.append(
                    {
                        "domain": "ai_think",
                        "pillar": pillar,
                        "level": 2,
                        "prompt": f"think: {p_short} ->",
                        "think": th_clean,
                        "ans": ans_distilled,
                    }
                )

    ver_path = os.path.join(out_dir, "verifiable_frontier.jsonl")
    thk_path = os.path.join(out_dir, "ai_reasoning_think.jsonl")
    rep_path = os.path.join(out_dir, "curation_report.json")

    with open(ver_path, "w", encoding="utf-8") as f:
        for r in verifiable_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    with open(thk_path, "w", encoding="utf-8") as f:
        for r in reasoning_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    report = {
        "kept_total": len(verifiable_rows) + len(reasoning_rows),
        "kept_verifiable_frontier": len(verifiable_rows),
        "kept_ai_reasoning_think": len(reasoning_rows),
        "dropped_counts": dict(dropped),
        "top_pillars": dict(pillar_counts.most_common(20)),
    }
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 88)
    print("PHILOSOPHER 1.0 DATA CURATION SUMMARY")
    print("=" * 88)
    print(f"Saved clean curated datasets to: {out_dir}")
    print(f"  1. verifiable_frontier.jsonl : {len(verifiable_rows):4d} deterministic AI/logic/math challenges")
    print(f"  2. ai_reasoning_think.jsonl  : {len(reasoning_rows):4d} distilled AI/RSI/invention <think> traces")
    print(f"  TOTAL UNIQUE HIGH-SIGNAL     : {report['kept_total']:4d} episodes")
    print("\nFiltered out unuseful / duplicate raw rows:")
    for k, v in dropped.most_common():
        print(f"  - {k:34s}: {v:5d}")
    print("=" * 88)
    return report


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    curate_all(os.path.join(here, "data"))
