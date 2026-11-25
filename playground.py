#!/usr/bin/env python3
"""
Philosopher 1.0 — Interactive Block-Diffusion & Wreath-Gear Playground
======================================================================
Watch all 32 `<MASK>` supertokens sharpen in parallel across:
  Pass 0 (NCM Coarse Skeleton) -> TRM Pass 1 -> TRM Pass 2 -> TRM Pass 3!
Also inspects internal 8x8 Swap-Gears, 12-Hour 30° Clock-Dials, and Dead-Zone Gates!

Usage:
  python3 playground.py         # Launch interactive terminal playground
  python3 playground.py --demo  # Run non-interactive live showcase across all 3 domains
"""

import argparse
import math
import os
import torch
import torch.nn.functional as F

from philosopher.brain import BLOCK_LEN, SPECIALIST_NAMES, snap_signed_perm
from philosopher.rsi_engine import PhilosopherRSIManager, collate_prompts_only, decode_zones
from philosopher.sandbox_verifier import EvolutionaryChallenger, parse_and_verify_prompt
from run_rsi_session import print_scoreboard


def format_pass_draft(
    b_logits_1d: torch.Tensor,
    m_logits_1d: torch.Tensor,
    tok,
    pass_idx: int = 0,
    final_pass: bool = False,
) -> tuple[str, str, float, int]:
    probs_b = F.softmax(b_logits_1d, dim=-1)
    probs_m = F.softmax(m_logits_1d, dim=-1)
    conf_b, pred_b = probs_b.max(dim=-1)
    pred_m = probs_m.argmax(dim=-1)

    mask_id = tok.base_to_id["<MASK>"]
    active_mask = pred_b != 0
    mean_conf = float(conf_b[active_mask].mean().item() * 100.0) if active_mask.any() else 100.0
    q_frac = 0.70 if pass_idx == 0 else (0.50 if pass_idx == 1 else 0.25)
    dyn_lock = min(0.75, float(torch.quantile(conf_b[active_mask], q_frac).item())) if active_mask.any() else 0.75
    locked_cnt = int(((conf_b >= dyn_lock) & active_mask).sum().item())
    active_cnt = max(1, int(active_mask.sum().item()))

    if final_pass:
        vis_b = pred_b.cpu().tolist()
    else:
        vis_b = [
            int(pb) if (float(cb) >= dyn_lock or int(pb) == 0) else mask_id
            for pb, cb in zip(pred_b.cpu().tolist(), conf_b.cpu().tolist())
        ]
    vis_m = pred_m.cpu().tolist()
    think_s, ans_s = decode_zones(vis_b, vis_m, tok, show_masks=True)
    return think_s, ans_s, mean_conf, int(round(100.0 * locked_cnt / active_cnt))


def inspect_internal_gears(brain) -> str:
    # 1. Inspect Swap-Gear cumulative permutation matrix at end of prompt
    U = snap_signed_perm(brain.last_U_swap[:1])[0].cpu()
    d_sw = U.shape[1]
    perm_map = []
    moved = 0
    for col in range(d_sw):
        row = int(U[:, col].abs().argmax().item())
        sgn = "+" if float(U[row, col].item()) >= 0 else "-"
        if row != col or sgn == "-":
            moved += 1
        perm_map.append(f"B{col}->{sgn}B{row}")

    # 2. Inspect 4x 2D Unit-Circle Clock-Dials (snapped to 12-hour 30° ticks)
    clk_xy = brain.last_clk_final[0].cpu()  # [4, 2]
    dial_strs = []
    for d_i in range(clk_xy.shape[0]):
        ang = math.atan2(float(clk_xy[d_i, 1]), float(clk_xy[d_i, 0]))
        deg = (math.degrees(ang)) % 360.0
        hr = int(round(deg / 30.0)) % 12
        dial_strs.append(f"D{d_i}={hr:2d}h({deg:3.0f}°)")

    # 3. Inspect Dead-Zone Gate silence rate across real prompt tokens
    real_mask = brain.last_not_pad[0, :, 0] > 0.5
    b2_real = brain.last_b2[0, real_mask, 0]
    quiet_pct = float((b2_real == 0.0).float().mean().item() * 100.0) if b2_real.numel() > 0 else 0.0

    # 4. Inspect Kimi K3.1 Stable LatentMoE + 44-Layer Task Specialists + AttnRes!
    loads = brain.trm_moe.last_loads.tolist()
    moe_s = " ".join(f"E{i}:{v*50.0:3.0f}%" for i, v in enumerate(loads))
    sp_w = brain.specialist_bank.last_spec_weights.tolist()
    sp_s = " ".join(f"{SPECIALIST_NAMES[i]}:{v*100.0:3.0f}%" for i, v in enumerate(sp_w) if v > 0.01)
    ar_w = F.softmax(brain.attn_res_logits[3, :4], dim=0).detach().cpu().tolist()
    ar_s = "/".join(f"{w:.2f}" for w in ar_w)

    return (
        f"  [Wreath L20   ] Swap-Boxes ({moved}/{d_sw} shifted): {' '.join(perm_map[:5])} ...\n"
        f"  [Clock & Gate ] 30° Dials: {' | '.join(dial_strs)} | Dead-Zone Silenced: {quiet_pct:4.1f}%\n"
        f"  [40L + 4 Spec ] Specialists [{sp_s}] | LatentMoE [{moe_s}] | AttnRes [{ar_s}]"
    )


def solve_and_visualize(prompt: str, mgr: PhilosopherRSIManager, brain, tok, state: dict, known_gold: str | None = None):
    brain.eval()
    clk_dz, box_dz, snap_s, snap_c = mgr.get_eval_flags(state)

    pb, pm = collate_prompts_only([prompt], tok, mgr.device)
    is_pr = prompt.strip().startswith(("think:", "reason:"))
    with torch.no_grad():
        b_logs, m_logs, _, _ = brain(
            pb, pm, clk_dz=clk_dz, box_dz=box_dz, snap_swap=snap_s, snap_clk=snap_c, explore_temp=0.0, prose_mode=is_pr
        )

    print("\n" + "-" * 98)
    print(f"PROMPT : {prompt}")
    print("-" * 98)
    print(f"  {'Start (40 Masks)':<18} | {'▪' * 28}  ||  {'▪' * 12}")

    pass_names = [
        "Pass 0 (NCM Plan)",
        "Pass 1 (TRM Step)",
        "Pass 2 (TRM Step)",
        "Pass 3 (Final)   ",
    ]
    final_ans = ""
    final_think = ""
    for k_idx, p_name in enumerate(pass_names):
        is_last = k_idx == 3
        th_s, an_s, m_conf, lk_pct = format_pass_draft(
            b_logs[k_idx][0], m_logs[k_idx][0], tok, pass_idx=k_idx, final_pass=is_last
        )
        if is_last:
            final_ans = an_s
            final_think = th_s
        th_disp = th_s if len(th_s) <= 52 else (th_s[:49] + "...")
        an_disp = an_s if len(an_s) <= 16 else (an_s[:13] + "...")
        print(f"  {p_name:<18} | {th_disp:<52} || <ans> {an_disp:<16} </ans>  ({m_conf:4.1f}%, lock {lk_pct:3d}%)")

    if len(final_think) > 52:
        print(f"  [Full <think>]     {final_think}")

    print(inspect_internal_gears(brain))
    ok, gold = parse_and_verify_prompt(prompt, final_ans, known_gold=known_gold)
    if gold:
        verdict = "VERIFIED CORRECT ✓" if ok else f"MISMATCH ✗ (Sandbox Expected: {gold})"
        print(f"  [Sandbox Verifier] Predicted: {final_ans}  -->  {verdict}")
    else:
        print(f"  [Output Answer]    {final_ans}")
    print("-" * 92)


def run_demo(mgr: PhilosopherRSIManager):
    brain, tok, state = mgr.load_or_init_system()
    lv = max(1, int(state.get("current_level", 1)))
    challenger = EvolutionaryChallenger(seed=777 + int(state.get("session", 0)) * 19)

    print("=" * 92)
    print(
        f"PHILOSOPHER 1.0 LIVE DEMO | Session #{state['session']} Brain | "
        f"Active Frontier: Level {lv} | Vault: {state['vault_size']} traces"
    )
    print("=" * 92)

    # Showcase 1 puzzle from each domain (`induce:`, `py:`, `story:`, plus an AI/RSI `<think>` prompt!)
    seen_domains: set[str] = set()
    demo_items = []
    while len(demo_items) < 3:
        ch = challenger.propose_challenge(level=min(lv, 2))
        if ch.domain not in seen_domains:
            seen_domains.add(ch.domain)
            demo_items.append((ch.prompt, ch.gold_ans))

    ai_think_all, _ = mgr._load_curated_corpora()
    if ai_think_all:
        demo_items.append((ai_think_all[-1]["prompt"], None))

    for p_str, g_ans in demo_items:
        solve_and_visualize(p_str, mgr, brain, tok, state, known_gold=g_ans)


def run_repl(mgr: PhilosopherRSIManager):
    brain, tok, state = mgr.load_or_init_system()
    challenger = EvolutionaryChallenger(seed=4040 + int(state.get("session", 0)))

    print("\n" + "=" * 92)
    print("WELCOME TO PHILOSOPHER 1.0 — INTERACTIVE BLOCK-DIFFUSION & RSI PLAYGROUND")
    print("=" * 92)
    print("Commands you can type:")
    print("  challenge        -> Challenger invents a puzzle at Philosopher's current Level")
    print("  challenge 2      -> Challenger invents a Level 2 (or 1, 3) puzzle")
    print("  improve          -> Run 1 live Challenger-Solver RSI self-improvement session right now!")
    print("  stats            -> View session-by-session RSI scoreboard & Knowledge Vault size")
    print("  quit             -> Exit playground")
    print("\nOr type your own custom puzzle directly! Examples:")
    print("  py: set a= 2 ; set b= 5 ; set c= 1 ; set clk= 11 ; tick clk +1 ; if clk even then swap a,b ->")
    print("  story: set Ava= 4 , set Ben= 7 , set Zoe= 2 , set dial= 5. tick dial +1. swap Ava,Ben. query ->")
    print("  induce: [1,2,3] -> [3,2,1] ; [4,0,6] -> [6,0,4] ; [2,5,7] ->")
    print("=" * 92)

    while True:
        try:
            cmd = input(f"\n[Philosopher L{state['current_level']}] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not cmd:
            continue
        if cmd.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        if cmd.lower() in ("stats", "status", "score"):
            print_scoreboard(mgr.state_path)
            continue
        if cmd.lower() in ("improve", "rsi", "train"):
            mgr.run_session()
            brain, tok, state = mgr.load_or_init_system()
            continue
        if cmd.lower().startswith("challenge"):
            parts = cmd.split()
            lv = int(parts[1]) if (len(parts) > 1 and parts[1].isdigit()) else int(state["current_level"])
            ch = challenger.propose_challenge(level=max(1, min(4, lv)), stumper_genes=state.get("stumper_genes"))
            solve_and_visualize(ch.prompt, mgr, brain, tok, state, known_gold=ch.gold_ans)
            continue

        solve_and_visualize(cmd, mgr, brain, tok, state)


def main():
    parser = argparse.ArgumentParser(description="Philosopher 1.0 Interactive Playground")
    parser.add_argument("--demo", action="store_true", help="Run a non-interactive showcase across all 3 domains")
    args = parser.parse_args()

    root_dir = os.path.dirname(os.path.abspath(__file__))
    mgr = PhilosopherRSIManager(root_dir)

    if args.demo:
        run_demo(mgr)
    else:
        run_repl(mgr)


if __name__ == "__main__":
    main()
