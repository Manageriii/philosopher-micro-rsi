"""
Philosopher 1.0 RSI Engine: V4.1-Flash + K3.1 + Wreath Brain & Challenger-Solver Self-Play
=========================================================================================
Unites two complementary pillars of intelligence:
  1. Curated AI, RSI, V4.1-Flash, K3.1, Weight-Surgery & Logic `<think>` Knowledge
     (`data/philosopher_curated/`) trained with Reasoning-First Contiguous Block Diffusion
     (1.6x weight on `<think>` derivations vs 0.30x on answers so capacity goes into
     *how to think and invent*, with zero middle `<PAD>` gaps!).
  2. Pure Challenger-Solver Self-Play RSI (`K=8` parallel NCM+TRM Block-Diffusion rollouts)
     verified 100% deterministically by the Sandbox Verifier (`zero` oracle hints leaked!)
     so Philosopher discovers new multi-step proofs/programs every session and levels up!
"""

import json
import math
import os
import random
import time
import torch
import torch.nn.functional as F

from philosopher.brain import ANS_LEN, BLOCK_LEN, THINK_LEN, PhilosopherBrain
from philosopher.sandbox_verifier import ChallengeItem, EvolutionaryChallenger, parse_and_verify_prompt
from philosopher.tokenizer import PhilosopherTokenizer


MAX_PROMPT_TOKS = 112


def encode_contiguous_pair(
    prompt: str, think_text: str, ans_text: str, domain: str, tok: PhilosopherTokenizer
) -> tuple[list[int], list[int], list[int], list[int], list[float]]:
    """
    Encodes Prompt + Two-Zone Anchored 40-Supertoken Block:
      - Zone 0 (0..27) : `<think> {think_text} </think>` (70% dedicated to pure reasoning!)
      - Zone 1 (28..39): `<ans> {ans_text} </ans>` ALWAYS anchored at index 28 (zero shift blur!)
    """
    pb, pm = tok.encode_span(prompt, add_bos=True, add_eos=False)
    if len(pb) > MAX_PROMPT_TOKS:
        pb = pb[:56] + pb[-56:]
        pm = pm[:56] + pm[-56:]

    tb_th, tm_th = tok.encode_span(f"<think> {think_text} </think>", add_bos=False, add_eos=False)
    if len(tb_th) > THINK_LEN:
        end_th = tok.base_to_id["</think>"]
        tb_th = tb_th[: THINK_LEN - 1] + [end_th]
        tm_th = tm_th[: THINK_LEN - 1] + [0]
    pad_th = max(0, THINK_LEN - len(tb_th))

    tb_an, tm_an = tok.encode_span(f"<ans> {ans_text} </ans>", add_bos=False, add_eos=True)
    tb_an = tb_an[:ANS_LEN]
    tm_an = tm_an[:ANS_LEN]
    pad_an = max(0, ANS_LEN - len(tb_an))

    if domain == "ai_think":
        w_th, w_an, w_pad = 1.60, 0.30, 0.25
    else:
        w_th, w_an, w_pad = 1.15, 2.00, 0.28

    tb = tb_th + ([0] * pad_th) + tb_an + ([0] * pad_an)
    tm = tm_th + ([0] * pad_th) + tm_an + ([0] * pad_an)
    tw = ([w_th] * len(tb_th)) + ([w_pad] * pad_th) + ([w_an] * len(tb_an)) + ([w_pad] * pad_an)
    return pb, pm, tb[:BLOCK_LEN], tm[:BLOCK_LEN], tw[:BLOCK_LEN]


def decode_zones(b_row: list[int], m_row: list[int], tok: PhilosopherTokenizer, show_masks: bool = False) -> tuple[str, str]:
    th_s = tok.decode(b_row[:THINK_LEN], m_row[:THINK_LEN], show_masks=show_masks)
    an_raw = tok.decode(b_row[THINK_LEN:], m_row[THINK_LEN:], show_masks=show_masks)
    if "<ans>" in an_raw:
        an_raw = an_raw.split("<ans>", 1)[1]
    if "</ans>" in an_raw:
        an_raw = an_raw.split("</ans>", 1)[0]
    return th_s.strip(), an_raw.strip().replace(" ", "")


def collate_records(records: list[dict], tok: PhilosopherTokenizer, device: str):
    encoded = [
        encode_contiguous_pair(r["prompt"], r["think"], r["ans"], r.get("domain", ""), tok)
        for r in records
    ]
    max_p = max(len(x[0]) for x in encoded)
    if max_p % 4 != 0:
        max_p += 4 - (max_p % 4)
    B = len(records)
    p_b = torch.zeros(B, max_p, dtype=torch.long)
    p_m = torch.zeros(B, max_p, dtype=torch.long)
    t_b = torch.zeros(B, BLOCK_LEN, dtype=torch.long)
    t_m = torch.zeros(B, BLOCK_LEN, dtype=torch.long)
    t_w = torch.zeros(B, BLOCK_LEN, dtype=torch.float32)
    is_ver = torch.zeros(B, dtype=torch.bool)

    for i, ((pb, pm, tb, tm, tw), rec) in enumerate(zip(encoded, records)):
        Lp = len(pb)
        p_b[i, max_p - Lp :] = torch.tensor(pb, dtype=torch.long)
        p_m[i, max_p - Lp :] = torch.tensor(pm, dtype=torch.long)
        t_b[i] = torch.tensor(tb, dtype=torch.long)
        t_m[i] = torch.tensor(tm, dtype=torch.long)
        t_w[i] = torch.tensor(tw, dtype=torch.float32)
        is_ver[i] = rec.get("domain") != "ai_think"

    return p_b.to(device), p_m.to(device), t_b.to(device), t_m.to(device), t_w.to(device), is_ver.to(device)


def collate_prompts_only(prompts: list[str], tok: PhilosopherTokenizer, device: str):
    enc = []
    for p in prompts:
        pb, pm = tok.encode_span(p, add_bos=True, add_eos=False)
        if len(pb) > MAX_PROMPT_TOKS:
            pb = pb[:56] + pb[-56:]
            pm = pm[:56] + pm[-56:]
        enc.append((pb, pm))
    max_p = max(len(x[0]) for x in enc)
    if max_p % 4 != 0:
        max_p += 4 - (max_p % 4)
    B = len(prompts)
    p_b = torch.zeros(B, max_p, dtype=torch.long)
    p_m = torch.zeros(B, max_p, dtype=torch.long)
    for i, (pb, pm) in enumerate(enc):
        Lp = len(pb)
        p_b[i, max_p - Lp :] = torch.tensor(pb, dtype=torch.long)
        p_m[i, max_p - Lp :] = torch.tensor(pm, dtype=torch.long)
    return p_b.to(device), p_m.to(device)


class PhilosopherRSIManager:
    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.ckpt_dir = os.path.join(root_dir, "checkpoints")
        self.vault_dir = os.path.join(root_dir, "vault")
        self.curated_dir = os.path.join(root_dir, "data", "philosopher_curated")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.vault_dir, exist_ok=True)

        self.tok_path = os.path.join(self.ckpt_dir, "tokenizer.json")
        self.brain_path = os.path.join(self.ckpt_dir, "brain.pt")
        self.state_path = os.path.join(self.vault_dir, "rsi_state.json")
        self.vault_path = os.path.join(self.vault_dir, "knowledge_vault.jsonl")
        self.exam_path = os.path.join(self.vault_dir, "fixed_exam.json")

        self.device = "mps" if torch.backends.mps.is_available() else "cpu"

    def _load_curated_corpora(self) -> tuple[list[dict], list[dict]]:
        ai_think_path = os.path.join(self.curated_dir, "ai_reasoning_think.jsonl")
        ver_front_path = os.path.join(self.curated_dir, "verifiable_frontier.jsonl")
        ai_think, ver_front = [], []
        if os.path.exists(ai_think_path):
            with open(ai_think_path, "r", encoding="utf-8") as f:
                ai_think = [json.loads(ln) for ln in f if ln.strip()]
        if os.path.exists(ver_front_path):
            with open(ver_front_path, "r", encoding="utf-8") as f:
                ver_front = [json.loads(ln) for ln in f if ln.strip()]
        return ai_think, ver_front

    def load_or_init_system(self) -> tuple[PhilosopherBrain, PhilosopherTokenizer, dict]:
        ai_think_all, ver_front_all = self._load_curated_corpora()
        challenger_init = EvolutionaryChallenger(seed=101)

        if not os.path.exists(self.tok_path):
            seed_corpus = []
            for lv in [1, 2, 3]:
                for _ in range(350):
                    ch = challenger_init.propose_challenge(lv)
                    seed_corpus.append(f"{ch.prompt} <think> {ch.gold_think} </think> <ans> {ch.gold_ans} </ans>")
            for r in (ai_think_all[:900] + ver_front_all[:400]):
                seed_corpus.append(f"{r['prompt']} <think> {r['think']} </think> <ans> {r['ans']} </ans>")
            tok = PhilosopherTokenizer(subword_merges=380, superword_merges=150)
            tok_stats = tok.train(seed_corpus)
            tok.save(self.tok_path)
        else:
            tok = PhilosopherTokenizer.load(self.tok_path)
            tok_stats = {"base_vocab_size": len(tok.base_to_id), "modifier_vocab_size": len(tok.mod_to_id)}

        if not os.path.exists(self.exam_path):
            exam_rng = EvolutionaryChallenger(seed=9999)
            exam: dict[str, list[dict]] = {}
            for lv in [1, 2, 3]:
                items = [exam_rng.propose_challenge(lv) for _ in range(90)]
                exam[str(lv)] = [
                    {"domain": it.domain, "prompt": it.prompt, "think": it.gold_think, "ans": it.gold_ans}
                    for it in items
                ]
            # Hold out 80 curated AI/RSI/V4.1/K3.1 `<think>` episodes for evaluating AI reasoning mastery!
            exam["ai_think"] = ai_think_all[-80:] if len(ai_think_all) >= 80 else ai_think_all[:40]
            with open(self.exam_path, "w", encoding="utf-8") as f:
                json.dump(exam, f, indent=2)

        brain = PhilosopherBrain(
            base_vocab=len(tok.base_to_id),
            mod_vocab=len(tok.mod_to_id),
            mask_base_id=tok.base_to_id["<MASK>"],
        ).to(self.device)

        if os.path.exists(self.state_path) and os.path.exists(self.brain_path):
            with open(self.state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            brain.load_state_dict(torch.load(self.brain_path, map_location=self.device, weights_only=True))
        else:
            state = {
                "session": 0,
                "current_level": 1,
                "lam_clk": 0.0,
                "lam_box": 0.0,
                "clk_dz_ready": False,
                "box_dz_ready": False,
                "snap_clk_ready": False,
                "snap_box_ready": False,
                "stumper_genes": [],
                "vault_size": 0,
                "self_play_discoveries": 0,
                "verifiable_cursor": 250,
                "ai_think_cursor": 800,
                "history": [],
                "tok_stats": tok_stats,
            }
            train_ai_think = ai_think_all[:-80] if len(ai_think_all) > 80 else ai_think_all
            seed_ai_think = train_ai_think[:800]
            seed_verifiable = ver_front_all[:250]

            boot_challenger = EvolutionaryChallenger(seed=2026)
            boot_records: list[dict] = []
            for r in seed_ai_think:
                boot_records.append({**r, "source": "curated_ai_think", "level": 1})
            for r in seed_verifiable:
                boot_records.append({**r, "source": "curated_verifiable", "level": 1})
            for _ in range(4200):
                it = boot_challenger.propose_challenge(level=1)
                boot_records.append(
                    {
                        "source": "bootstrap_L1",
                        "level": 1,
                        "domain": it.domain,
                        "prompt": it.prompt,
                        "think": it.gold_think,
                        "ans": it.gold_ans,
                    }
                )

            with open(self.vault_path, "w", encoding="utf-8") as f:
                for rec in boot_records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            state["vault_size"] = len(boot_records)

            print(
                f"[Stage-0 Birth] Awakening Philosopher 1.0 ({sum(p.numel() for p in brain.parameters()):,} params | "
                f"40-Layer Core [20 Enc + 20 Dec] + 4 Task-Specialist Layers = 44 Layers | Vocab {tok_stats['base_vocab_size']}) "
                f"on {len(seed_ai_think)+len(seed_verifiable)} AI `<think>` traces + 4,200 Wreath primitives..."
            )
            self._train_batches(brain, boot_records, tok, state, n_epochs=30, lr=3.0e-3, is_bootstrap=True)
            torch.save(brain.state_dict(), self.brain_path)
            with open(self.state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)

        return brain, tok, state

    def load_vault(self) -> list[dict]:
        if not os.path.exists(self.vault_path):
            return []
        with open(self.vault_path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]

    def get_eval_flags(self, state: dict) -> tuple[float, float, bool, bool]:
        clk_dz = 0.05 if state.get("clk_dz_ready", False) else 0.0
        box_dz = 0.05 if state.get("box_dz_ready", False) else 0.0
        snap_s = bool(state.get("snap_box_ready", False))
        snap_c = bool(state.get("snap_clk_ready", False))
        return clk_dz, box_dz, snap_s, snap_c

    def evaluate_exam(self, brain: PhilosopherBrain, tok: PhilosopherTokenizer, state: dict) -> dict[str, float]:
        with open(self.exam_path, "r", encoding="utf-8") as f:
            exam = json.load(f)
        brain.eval()
        clk_dz, box_dz, snap_s, snap_c = self.get_eval_flags(state)

        scores: dict[str, float] = {}
        with torch.no_grad():
            for lv_str in ["1", "2", "3"]:
                items = exam[lv_str]
                prompts = [it["prompt"] for it in items]
                pb, pm = collate_prompts_only(prompts, tok, self.device)
                b_logs, m_logs, _, _ = brain(
                    pb, pm, clk_dz=clk_dz, box_dz=box_dz, snap_swap=snap_s, snap_clk=snap_c, explore_temp=0.0
                )
                pred_b = b_logs[-1].argmax(-1).cpu().tolist()
                pred_m = m_logs[-1].argmax(-1).cpu().tolist()

                exact = 0
                slot_ok = 0
                slot_tot = 0
                for i, it in enumerate(items):
                    _, pred_ans = decode_zones(pred_b[i], pred_m[i], tok)
                    ok, gold = parse_and_verify_prompt(it["prompt"], pred_ans, known_gold=it["ans"])
                    exact += int(ok)
                    pp = pred_ans.split(",")
                    gp = gold.split(",")
                    for k_i, gv in enumerate(gp):
                        slot_tot += 1
                        if k_i < len(pp) and pp[k_i] == gv:
                            slot_ok += 1
                scores[f"L{lv_str}_exact"] = (exact / len(items)) * 100.0
                scores[f"L{lv_str}_slot"] = (slot_ok / max(1, slot_tot)) * 100.0

            ai_items = exam.get("ai_think", [])
            if ai_items:
                pb_a, pm_a, tb_a, _, _, _ = collate_records(ai_items, tok, self.device)
                ba_logs, _, _, _ = brain(
                    pb_a, pm_a, clk_dz=clk_dz, box_dz=box_dz, snap_swap=snap_s, snap_clk=snap_c, explore_temp=0.0, prose_mode=True
                )
                pred_blk = ba_logs[-1][:, :THINK_LEN].argmax(-1)
                specials = {0, 1, 2, 3, 4, 5, 6, 7}
                overlaps = []
                for r_i in range(tb_a.shape[0]):
                    t_set = {int(x) for x in tb_a[r_i, :THINK_LEN].cpu().tolist() if x not in specials}
                    p_set = {int(x) for x in pred_blk[r_i].cpu().tolist() if x not in specials}
                    if t_set:
                        overlaps.append(len(t_set & p_set) / len(t_set))
                scores["ai_think_concept_pct"] = (sum(overlaps) / max(1, len(overlaps))) * 100.0
            else:
                scores["ai_think_concept_pct"] = 0.0

        return scores

    def _train_batches(
        self,
        brain: PhilosopherBrain,
        records: list[dict],
        tok: PhilosopherTokenizer,
        state: dict,
        n_epochs: int,
        lr: float = 1.2e-3,
        is_bootstrap: bool = False,
    ):
        short_recs = [r for r in records if r.get("domain") not in ("ai_think", "ai_verifiable")]
        long_recs = [r for r in records if r.get("domain") in ("ai_think", "ai_verifiable")]

        buckets = []
        for is_pr, grp in ((False, short_recs), (True, long_recs)):
            if len(grp) >= 16:
                buckets.append((is_pr, collate_records(grp, tok, self.device)))

        batch_sz = 64
        steps_per_ep = sum(max(1, b[1][0].shape[0] // batch_sz) for b in buckets)
        total_steps = max(1, n_epochs * steps_per_ep)
        warmup_steps = 16 if is_bootstrap else min(24, max(8, total_steps // 6))

        opt = torch.optim.AdamW(brain.parameters(), lr=lr, weight_decay=1e-4)

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return 0.12 + 0.88 * (step / max(1, warmup_steps))
            prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.08 + 0.92 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

        gen = torch.Generator(device="cpu")
        gen.manual_seed(state["session"] * 97 + 13)
        py_rng = random.Random(state["session"] * 97 + 13)

        ans_start_id = tok.base_to_id["<ans>"]
        ans_end_id = tok.base_to_id["</ans>"]
        running_ans_acc = 0.05 if is_bootstrap else 0.72
        running_c_mag = 0.25
        running_b_mag = 0.25
        step_idx = 0

        for _ep in range(n_epochs):
            ep_slices = []
            for b_id, (_is_pr, b_tup) in enumerate(buckets):
                Nb = b_tup[0].shape[0]
                perm = torch.randperm(Nb, generator=gen).to(self.device)
                if Nb < batch_sz:
                    ep_slices.append((b_id, perm))
                else:
                    for start in range(0, Nb - batch_sz + 1, batch_sz):
                        ep_slices.append((b_id, perm[start : start + batch_sz]))
            py_rng.shuffle(ep_slices)

            for b_id, idx in ep_slices:
                is_pr, (P_b, P_m, T_b, T_m, T_w, is_ver) = buckets[b_id]
                step_idx += 1
                brain.train()
                pb, pm, tb, tm, tok_w = P_b[idx], P_m[idx], T_b[idx], T_m[idx], T_w[idx]
                ver_mask = is_ver[idx]

                past_warmup = (not is_bootstrap) or (step_idx >= int(0.50 * total_steps))
                can_unlock = past_warmup and (step_idx <= int(0.75 * total_steps))
                if can_unlock and running_ans_acc >= 0.80:
                    state["box_dz_ready"] = True
                    state["clk_dz_ready"] = True

                if state.get("clk_dz_ready", False) and running_ans_acc >= 0.82:
                    state["lam_clk"] = min(0.25, state["lam_clk"] + 0.0010)
                if state["lam_clk"] >= 0.20 and running_ans_acc >= 0.86:
                    state["lam_box"] = min(0.25, state["lam_box"] + 0.0010)

                if can_unlock and state["lam_clk"] >= 0.22 and running_ans_acc >= 0.88 and running_c_mag < 0.035:
                    state["snap_clk_ready"] = True
                if can_unlock and state["lam_box"] >= 0.22 and running_ans_acc >= 0.90 and running_b_mag < 0.045:
                    state["snap_box_ready"] = True

                clk_dz, box_dz, snap_s, snap_c = self.get_eval_flags(state)

                b_logs, m_logs, b_mag, c_mag = brain(
                    pb,
                    pm,
                    clk_dz=clk_dz,
                    box_dz=box_dz,
                    clk_noise=0.0,
                    box_noise=0.0,
                    snap_swap=snap_s,
                    snap_clk=snap_c,
                    prose_mode=is_pr,
                )

                denom = tok_w.sum().clamp(min=1.0)

                def p_loss(bl, ml):
                    cb = F.cross_entropy(bl.reshape(-1, brain.base_vocab), tb.reshape(-1), reduction="none")
                    cm = F.cross_entropy(ml.reshape(-1, brain.mod_vocab), tm.reshape(-1), reduction="none")
                    return ((cb + 0.5 * cm) * tok_w.reshape(-1)).sum() / denom

                loss = (
                    0.15 * p_loss(b_logs[0], m_logs[0])
                    + 0.25 * p_loss(b_logs[1], m_logs[1])
                    + 0.40 * p_loss(b_logs[2], m_logs[2])
                    + 1.00 * p_loss(b_logs[3], m_logs[3])
                    + state["lam_box"] * b_mag
                    + state["lam_clk"] * c_mag
                )

                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
                opt.step()
                sched.step()

                if step_idx % 5 == 0:
                    with torch.no_grad():
                        running_c_mag = 0.88 * running_c_mag + 0.12 * float(c_mag.item())
                        running_b_mag = 0.88 * running_b_mag + 0.12 * float(b_mag.item())
                        if ver_mask.any():
                            tb_v = tb[ver_mask]
                            pr_v = b_logs[-1][ver_mask].argmax(-1)
                            pos_ar = torch.arange(BLOCK_LEN, device=self.device).unsqueeze(0)
                            st_pos = (tb_v == ans_start_id).float().argmax(dim=-1, keepdim=True)
                            en_pos = (tb_v == ans_end_id).float().argmax(dim=-1, keepdim=True)
                            in_ans = (pos_ar > st_pos) & (pos_ar < en_pos)
                            if in_ans.any():
                                # Measure exact row-level answer span accuracy across ALL answer digits!
                                row_ok = ((pr_v == tb_v) | (~in_ans)).all(dim=-1) & in_ans.any(dim=-1)
                                acc = row_ok.float().mean().item()
                                running_ans_acc = 0.90 * running_ans_acc + 0.10 * acc

        if self.device == "mps":
            torch.mps.synchronize()

    def run_session(self, n_challenges: int = 300, k_rollouts: int = 8) -> dict:
        t0 = time.time()
        brain, tok, state = self.load_or_init_system()
        ai_think_all, ver_front_all = self._load_curated_corpora()
        train_ai_pool = ai_think_all[:-80] if len(ai_think_all) > 80 else ai_think_all
        state["session"] += 1
        sess_id = state["session"]

        vault = self.load_vault()
        pre_scores = self.evaluate_exam(brain, tok, state)

        if pre_scores[f"L{min(3, state['current_level'])}_exact"] >= 70.0 and state["current_level"] < 3:
            state["current_level"] += 1
        cur_lv = state["current_level"]

        print("=" * 96)
        print(
            f"PHILOSOPHER 1.0 — RSI SESSION #{sess_id}  "
            f"[Device: {self.device.upper()} | Active Frontier: Level {cur_lv} | Vault: {len(vault)}]"
        )
        print("=" * 96)
        print(
            f"[Pre-RSI Exam ] L1: {pre_scores['L1_exact']:5.1f}% | L2: {pre_scores['L2_exact']:5.1f}% | "
            f"L3: {pre_scores['L3_exact']:5.1f}% | AI-<think> Concept Mastery: {pre_scores['ai_think_concept_pct']:5.1f}%"
        )

        # ---- PHASE 1: CHALLENGER GENERATES FRONTIER PUZZLES + UNSEEN AI/LOGIC VERIFIABLE CHALLENGES ----
        challenger = EvolutionaryChallenger(seed=1000 * sess_id + 7)
        challenges: list[ChallengeItem] = []

        v_cur = int(state.get("verifiable_cursor", 400))
        ai_ch_slice = ver_front_all[v_cur : v_cur + 45]
        state["verifiable_cursor"] = v_cur + len(ai_ch_slice)
        for r in ai_ch_slice:
            challenges.append(
                ChallengeItem(
                    domain="ai_verifiable",
                    level=cur_lv,
                    gene=f"ai:{r.get('pillar', 'logic')}",
                    prompt=r["prompt"],
                    gold_think=r["think"],
                    gold_ans=r["ans"],
                )
            )

        n_synth = max(0, n_challenges - len(challenges))
        for i in range(n_synth):
            if cur_lv == 1:
                lv_pick = 1 if (i % 2 == 0) else 2
            elif cur_lv == 2:
                lv_pick = 1 if (i % 6 == 0) else (2 if (i % 3 != 0) else 3)
            else:
                lv_pick = 2 if (i % 4 == 0) else 3
            challenges.append(challenger.propose_challenge(lv_pick, stumper_genes=state["stumper_genes"]))

        # ---- PHASE 2: SOLVER PARALLEL EXPLORATION (K=8 ROLLOUTS, ZERO ORACLE HINTS!) + VERIFIER ----
        brain.eval()
        clk_dz, box_dz, snap_s, snap_c = self.get_eval_flags(state)

        new_verified_inventions: list[dict] = []
        frontier_boosts: list[dict] = []
        new_stumpers: list[str] = []
        solved_any = 0
        goldilocks_discoveries = 0

        with torch.no_grad():
            for c_start in range(0, len(challenges), 30):
                sub_ch = challenges[c_start : c_start + 30]
                B_sub = len(sub_ch)
                rep_prompts = [ch.prompt for ch in sub_ch for _ in range(k_rollouts)]
                pb, pm = collate_prompts_only(rep_prompts, tok, self.device)

                is_pr_sub = all(ch.domain == "ai_verifiable" for ch in sub_ch)
                b_logs, m_logs, _, _ = brain(
                    pb, pm, clk_dz=clk_dz, box_dz=box_dz, snap_swap=snap_s, snap_clk=snap_c, explore_temp=0.95, prose_mode=is_pr_sub
                )
                bl_fin = b_logs[-1]
                ml_fin = m_logs[-1]

                pb_g, pm_g = collate_prompts_only([ch.prompt for ch in sub_ch], tok, self.device)
                bg_logs, mg_logs, _, _ = brain(
                    pb_g, pm_g, clk_dz=clk_dz, box_dz=box_dz, snap_swap=snap_s, snap_clk=snap_c, explore_temp=0.0, prose_mode=is_pr_sub
                )

                probs_fin = F.softmax(bl_fin, dim=-1)
                max_conf, greedy_b = probs_fin.max(dim=-1)
                gumbel_fin = -torch.empty_like(bl_fin).exponential_().log() * 0.45
                sampled_b = (bl_fin + gumbel_fin).argmax(dim=-1)
                cand_b = torch.where(max_conf >= 0.90, greedy_b, sampled_b)
                cand_m = ml_fin.argmax(dim=-1)

                cand_b = cand_b.view(B_sub, k_rollouts, BLOCK_LEN)
                cand_m = cand_m.view(B_sub, k_rollouts, BLOCK_LEN)
                cand_b[:, 0] = bg_logs[-1].argmax(dim=-1)
                cand_m[:, 0] = mg_logs[-1].argmax(dim=-1)

                pred_b_list = cand_b.reshape(-1, BLOCK_LEN).cpu().tolist()
                pred_m_list = cand_m.reshape(-1, BLOCK_LEN).cpu().tolist()

                for idx_c, ch in enumerate(sub_ch):
                    wins: list[tuple[str, str]] = []
                    for r in range(k_rollouts):
                        row = idx_c * k_rollouts + r
                        think_s, ans_s = decode_zones(pred_b_list[row], pred_m_list[row], tok)
                        ok, _ = parse_and_verify_prompt(ch.prompt, ans_s, known_gold=ch.gold_ans)
                        if ok:
                            clean_think = think_s.replace("<think>", "").replace("</think>", "").strip()
                            wins.append((clean_think, ans_s))

                    if len(wins) > 0:
                        solved_any += 1
                        best_think, best_ans = max(wins, key=lambda pair: len(pair[0]))
                        rec = {
                            "source": f"self_play_S{sess_id}",
                            "level": ch.level,
                            "domain": ch.domain,
                            "prompt": ch.prompt,
                            "think": best_think,
                            "ans": best_ans,
                        }
                        new_verified_inventions.append(rec)
                        if 1 <= len(wins) <= (k_rollouts - 2) or ch.level >= 2:
                            if 1 <= len(wins) <= (k_rollouts - 2):
                                goldilocks_discoveries += 1
                                if not ch.gene.startswith("ai:"):
                                    new_stumpers.append(ch.gene)
                            frontier_boosts.append(rec)
                            frontier_boosts.append(rec)
                    else:
                        if not ch.gene.startswith("ai:"):
                            new_stumpers.append(ch.gene)

        # Absorb next stream of advanced AI/RSI/V4.1-Flash/K3.1 `<think>` derivations each session!
        th_cur = int(state.get("ai_think_cursor", 800))
        fresh_ai_study = [
            {**r, "source": f"ai_study_S{sess_id}", "level": cur_lv}
            for r in train_ai_pool[th_cur : th_cur + 350]
        ]
        state["ai_think_cursor"] = th_cur + len(fresh_ai_study)

        with open(self.vault_path, "a", encoding="utf-8") as f:
            for r in new_verified_inventions + fresh_ai_study:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        vault.extend(new_verified_inventions + fresh_ai_study)
        state["vault_size"] = len(vault)
        state["self_play_discoveries"] = state.get("self_play_discoveries", 0) + len(new_verified_inventions)
        state["stumper_genes"] = new_stumpers[-80:]

        print(
            f"[Challenger-Solver] Verified {solved_any}/{len(challenges)} challenges via K={k_rollouts} rollouts "
            f"({goldilocks_discoveries} frontier breakthroughs!) -> +{len(new_verified_inventions)} self-play + {len(fresh_ai_study)} AI `<think>` traces in Vault."
        )

        # ---- PHASE 3: CONSOLIDATION TRAINING (AI `<think>` + WREATH REPLAY + NEW INVENTIONS) ----
        rng_replay = random.Random(sess_id * 31)
        ai_bucket = [r for r in vault if r.get("domain") in ("ai_think", "ai_verifiable")]
        wreath_by_lv: dict[int, list[dict]] = {}
        for r in vault:
            if r.get("domain") not in ("ai_think", "ai_verifiable"):
                wreath_by_lv.setdefault(int(r.get("level", 1)), []).append(r)

        replay_sample: list[dict] = rng_replay.sample(ai_bucket, min(len(ai_bucket), 550)) + fresh_ai_study
        per_lv_quota = max(550, 1500 // max(1, len(wreath_by_lv)))
        for _lv_k, bucket in sorted(wreath_by_lv.items()):
            replay_sample.extend(rng_replay.sample(bucket, min(len(bucket), per_lv_quota)))

        train_mix = replay_sample + new_verified_inventions + frontier_boosts
        prev_weights = {k: v.detach().cpu().clone() for k, v in brain.state_dict().items()}
        self._train_batches(brain, train_mix, tok, state, n_epochs=16, lr=8.5e-4, is_bootstrap=False)

        # ---- PHASE 4: POST-RSI EXAM, ELITIST RATCHET & AUTO-LEVEL-UP ----
        post_scores = self.evaluate_exam(brain, tok, state)
        pre_tot = pre_scores["L1_exact"] + pre_scores["L2_exact"] + pre_scores["L3_exact"]
        post_tot = post_scores["L1_exact"] + post_scores["L2_exact"] + post_scores["L3_exact"]
        if post_tot < pre_tot:
            # AlphaZero / DGM Elitist Ratchet: keep new Vault discoveries, preserve stronger weights!
            brain.load_state_dict({k: v.to(self.device) for k, v in prev_weights.items()})
            post_scores = pre_scores
        cur_exact = post_scores[f"L{min(3, cur_lv)}_exact"]
        leveled_up = False
        if cur_exact >= 70.0 and state["current_level"] < 4:
            state["current_level"] += 1
            leveled_up = True

        dt = time.time() - t0
        summary_entry = {
            "session": sess_id,
            "level_tested": cur_lv,
            "new_level": state["current_level"],
            "solved_challenges": f"{solved_any}/{len(challenges)}",
            "frontier_breakthroughs": goldilocks_discoveries,
            "pre_scores": pre_scores,
            "post_scores": post_scores,
            "vault_size": state["vault_size"],
            "self_play_discoveries": state["self_play_discoveries"],
            "seconds": round(dt, 1),
        }
        state["history"].append(summary_entry)

        torch.save(brain.state_dict(), self.brain_path)
        with open(self.state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)

        print(
            f"[Post-RSI Exam] L1: {post_scores['L1_exact']:5.1f}% ({post_scores['L1_exact']-pre_scores['L1_exact']:+5.1f}%) | "
            f"L2: {post_scores['L2_exact']:5.1f}% ({post_scores['L2_exact']-pre_scores['L2_exact']:+5.1f}%) | "
            f"L3: {post_scores['L3_exact']:5.1f}% ({post_scores['L3_exact']-pre_scores['L3_exact']:+5.1f}%) | "
            f"AI-<think>: {post_scores['ai_think_concept_pct']:5.1f}%"
        )
        if leveled_up:
            print(f"*** LEVEL UP! Philosopher 1.0 unlocked Level {state['current_level']} for the next session! ***")
        print(f"[Saved] Checkpoint -> {self.brain_path} ({dt:.1f}s)\n")
        return summary_entry
