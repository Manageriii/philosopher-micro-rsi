#!/usr/bin/env python3
"""
Philosopher 1.0 — Recursive Self-Improvement (RSI) Session Runner
=================================================================
Usage:
  python3 run_rsi_session.py               # Run 1 Challenger-Solver self-play session
  python3 run_rsi_session.py --sessions 3  # Run 3 back-to-back RSI sessions
  python3 run_rsi_session.py --status      # View current Level, Vault size & RSI history
"""

import argparse
import json
import os

from philosopher.rsi_engine import PhilosopherRSIManager


def print_scoreboard(state_path: str):
    if not os.path.exists(state_path):
        print("No saved Philosopher 1.0 brain yet! Run `python3 run_rsi_session.py` to awaken it.")
        return

    with open(state_path, "r", encoding="utf-8") as f:
        st = json.load(f)

    print("\n" + "=" * 92)
    print(
        f"PHILOSOPHER 1.0 SCOREBOARD | Sessions Completed: {st['session']} | "
        f"Current Frontier: Level {st['current_level']} | Vault: {st['vault_size']} traces "
        f"({st.get('self_play_discoveries', 0)} self-discovered)"
    )
    print("-" * 102)
    print(
        f"{'Session':<10} | {'Frontier':<9} | {'Level 1':<15} | {'Level 2':<15} | "
        f"{'Level 3':<15} | {'AI-<think>':<10} | {'Solved'}"
    )
    print("-" * 102)
    for idx, h in enumerate(st.get("history", [])):
        if idx == 0:
            p0 = h["pre_scores"]
            print(
                f"{'Birth(S0)':<10} | {'Level 1':<9} | "
                f"{p0['L1_exact']:5.1f}%          | "
                f"{p0['L2_exact']:5.1f}%          | "
                f"{p0['L3_exact']:5.1f}%          | "
                f"{p0.get('ai_think_concept_pct', 0.0):5.1f}%     | {'(foundational)'}"
            )
        ps = h["post_scores"]
        pr = h["pre_scores"]
        d1 = ps["L1_exact"] - pr["L1_exact"]
        d2 = ps["L2_exact"] - pr["L2_exact"]
        d3 = ps["L3_exact"] - pr["L3_exact"]
        lv_tag = f"L{h['level_tested']}->L{h['new_level']}" if h["new_level"] > h["level_tested"] else f"Level {h['level_tested']}"
        print(
            f"{'#' + str(h['session']):<10} | {lv_tag:<9} | "
            f"{ps['L1_exact']:5.1f}% ({d1:+5.1f}%) | "
            f"{ps['L2_exact']:5.1f}% ({d2:+5.1f}%) | "
            f"{ps['L3_exact']:5.1f}% ({d3:+5.1f}%) | "
            f"{ps.get('ai_think_concept_pct', 0.0):5.1f}%     | "
            f"{h.get('solved_challenges', '-')}"
        )
    print("=" * 102 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run Philosopher 1.0 Challenger-Solver RSI sessions.")
    parser.add_argument("--sessions", type=int, default=1, help="Number of RSI self-play sessions to run (default: 1)")
    parser.add_argument("--challenges", type=int, default=240, help="Challenger puzzles per session (default: 240)")
    parser.add_argument("--rollouts", type=int, default=8, help="Parallel Solver rollouts per puzzle (default: 8)")
    parser.add_argument("--status", action="store_true", help="Show persistent RSI scoreboard without training")
    args = parser.parse_args()

    root_dir = os.path.dirname(os.path.abspath(__file__))
    mgr = PhilosopherRSIManager(root_dir)

    if args.status:
        print_scoreboard(mgr.state_path)
        return

    for _ in range(max(1, args.sessions)):
        mgr.run_session(n_challenges=args.challenges, k_rollouts=args.rollouts)

    print_scoreboard(mgr.state_path)


if __name__ == "__main__":
    main()
