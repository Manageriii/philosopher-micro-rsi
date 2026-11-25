"""
Philosopher 1.0 Sandbox Verifier & Evolutionary Challenger Engine
=================================================================
100% deterministic, zero-`eval()` sandbox that:
  1. Generates open-ended compositional challenges across 3 reasoning domains
     (`induce:`, `py:`, `story:`) scaled by `level` (Level 1 -> Level 2 -> Level 3+...).
  2. Mutates/recombines "stumper genes" (rules/programs where Solver only passed 1-3 out
     of 4 attempts) so the Challenger actively targets the Solver's frontier blind spots!
  3. Deterministically parses & verifies ANY prompt + candidate `<think>...<ans>...</ans>`
     so ZERO hallucinated answers can ever enter the Knowledge Vault!
"""

import json
import random
import re
from dataclasses import dataclass


@dataclass
class ChallengeItem:
    domain: str
    level: int
    gene: str
    prompt: str
    gold_think: str
    gold_ans: str


# Compositional Primitive Operators for Open-Ended Mystery-Rule Invention (`induce:`)
PRIMITIVE_RULES = {
    "rev": ("reverse", lambda a: [a[2], a[1], a[0]]),
    "rotL": ("rotL", lambda a: [a[1], a[2], a[0]]),
    "sw01": ("swap01", lambda a: [a[1], a[0], a[2]]),
    "sw12": ("swap12", lambda a: [a[0], a[2], a[1]]),
    "add1": ("+1m8", lambda a: [(x + 1) % 8 for x in a]),
    "add2": ("+2m8", lambda a: [(x + 2) % 8 for x in a]),
    "ev_sw": ("if0ev_sw12", lambda a: [a[0], a[2], a[1]] if a[0] % 2 == 0 else list(a)),
    "clk_sh": ("+a0m12", lambda a: [a[0], (a[1] + a[0]) % 12, (a[2] + a[0]) % 12]),
}

FILLERS = ["note ok.", "check pass.", "log idle.", "keep state.", "wait step."]


def apply_composed_rule(arr: list[int], ops: list[str]) -> list[int]:
    cur = list(arr)
    for op in ops:
        cur = PRIMITIVE_RULES[op][1](cur)
    return cur


def parse_and_verify_prompt(prompt: str, candidate_ans: str, known_gold: str | None = None) -> tuple[bool, str]:
    """
    Deterministically computes or verifies the exact answer for ANY `py:`, `story:`, or `induce:` prompt!
    Returns `(is_correct, exact_gold_answer)`.
    """
    cand = candidate_ans.strip().replace(" ", "")
    p = prompt.strip()

    if p.startswith("py:"):
        body = p[len("py:") :].split("->", 1)[0]
        stmts = [s.strip() for s in body.split(";") if s.strip()]
        st = {"a": 0, "b": 0, "c": 0, "clk": 0}
        for s in stmts:
            m_set = re.match(r"set\s+(a|b|c|clk)=\s*(-?\d+)", s)
            if m_set:
                st[m_set.group(1)] = int(m_set.group(2))
                continue
            m_tick = re.match(r"tick\s+clk\s+([+-]?\d+)", s)
            if m_tick:
                st["clk"] = (st["clk"] + int(m_tick.group(1))) % 12
                continue
            if s == "swap a,b":
                st["a"], st["b"] = st["b"], st["a"]
            elif s == "swap b,c":
                st["b"], st["c"] = st["c"], st["b"]
            elif s == "if clk even then swap a,b":
                if st["clk"] % 2 == 0:
                    st["a"], st["b"] = st["b"], st["a"]
            elif s == "if a even then swap b,c":
                if st["a"] % 2 == 0:
                    st["b"], st["c"] = st["c"], st["b"]
        gold = f"{st['a']},{st['b']},{st['c']},{st['clk']}"
        return (cand == gold), gold

    if p.startswith("story:"):
        body = p[len("story:") :].split("query", 1)[0]
        st = {"Ava": 0, "Ben": 0, "Zoe": 0, "dial": 0}
        for m_init in re.finditer(r"set\s+(Ava|Ben|Zoe|dial)=\s*(\d+)", body):
            st[m_init.group(1)] = int(m_init.group(2))
        # Split sentences after initial setup sentence
        clauses = [c.strip() for c in body.split(".") if c.strip()][1:]
        for c in clauses:
            mt = re.match(r"tick\s+dial\s+\+(\d+)", c)
            if mt:
                st["dial"] = (st["dial"] + int(mt.group(1))) % 12
            elif c == "swap Ava,Ben":
                st["Ava"], st["Ben"] = st["Ben"], st["Ava"]
            elif c == "swap Ben,Zoe":
                st["Ben"], st["Zoe"] = st["Zoe"], st["Ben"]
            elif c == "if Ava even then swap Ben,Zoe":
                if st["Ava"] % 2 == 0:
                    st["Ben"], st["Zoe"] = st["Zoe"], st["Ben"]
            elif c == "if dial even then swap Ava,Ben":
                if st["dial"] % 2 == 0:
                    st["Ava"], st["Ben"] = st["Ben"], st["Ava"]
        gold = f"{st['Ava']},{st['Ben']},{st['Zoe']},{st['dial']}"
        return (cand == gold), gold

    if p.startswith("reason:") and known_gold is not None:
        gold = known_gold.strip().replace(" ", "")
        try:
            g_obj = json.loads(known_gold)
            c_obj = json.loads(candidate_ans)
            return (c_obj == g_obj), gold
        except Exception:
            return (cand == gold), gold

    if p.startswith("induce:") and known_gold is not None:
        gold = known_gold.strip().replace(" ", "")
        return (cand == gold), gold

    # Fallback search over 1-, 2-, and 3-step compositions for custom user-typed `induce:` prompts!
    if p.startswith("induce:"):
        arrays = [[int(x) for x in m.split(",")] for m in re.findall(r"\[([\d,\s]+)\]", p)]
        if len(arrays) >= 5 and len(arrays) % 2 == 1:
            pairs = [(arrays[i], arrays[i + 1]) for i in range(0, len(arrays) - 1, 2)]
            test_in = arrays[-1]
            keys = list(PRIMITIVE_RULES.keys())
            candidates = (
                [[k] for k in keys]
                + [[k1, k2] for k1 in keys for k2 in keys]
                + [[k1, k2, k3] for k1 in keys for k2 in keys for k3 in keys]
            )
            for ops in candidates:
                if all(apply_composed_rule(inp, ops) == out for inp, out in pairs):
                    pred_out = apply_composed_rule(test_in, ops)
                    gold = f"{pred_out[0]},{pred_out[1]},{pred_out[2]}"
                    return (cand == gold), gold

    return False, (known_gold or "")


class EvolutionaryChallenger:
    """
    Challenger half of Philosopher 1.0's Self-Play Loop:
      - Generates challenges calibrated to `level` (Level 1 = 1-step rules & 2-4 op code;
        Level 2 = 2-step composed rules & conditional swaps; Level 3+ = multi-hop chains!).
      - Tracks `stumper_genes` discovered in previous sessions where the Solver struggled
        (1-3/4 pass rate) and mutates/breeds them to sharpen weak spots!
    """

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)
        self.prim_keys = list(PRIMITIVE_RULES.keys())

    def propose_challenge(self, level: int, stumper_genes: list[str] | None = None) -> ChallengeItem:
        # 40% of the time, mutate/breed a previous session's borderline stumper gene!
        use_mutation = bool(stumper_genes) and (self.rng.random() < 0.40)
        parent_gene = self.rng.choice(stumper_genes) if use_mutation else None

        if parent_gene and parent_gene.startswith("induce:"):
            domain = "induce"
        elif parent_gene and parent_gene.startswith("py:"):
            domain = "py"
        elif parent_gene and parent_gene.startswith("story:"):
            domain = "story"
        else:
            domain = self.rng.choices(["induce", "py", "story"], weights=[0.36, 0.34, 0.30])[0]

        if domain == "induce":
            return self._make_induce(level, parent_gene)
        elif domain == "py":
            return self._make_py(level, parent_gene)
        else:
            return self._make_story(level, parent_gene)

    def _make_induce(self, level: int, parent_gene: str | None) -> ChallengeItem:
        if parent_gene and ":" in parent_gene:
            base_ops = parent_gene.split(":", 1)[1].split("+")
            # Mutate one operator or append a composition at higher levels!
            ops = list(base_ops)
            if level >= 2 and len(ops) < min(3, level) and self.rng.random() < 0.5:
                ops.append(self.rng.choice(self.prim_keys))
            else:
                idx = self.rng.randrange(len(ops))
                ops[idx] = self.rng.choice(self.prim_keys)
        else:
            n_comp = 1 if level == 1 else (2 if level == 2 else self.rng.choice([2, min(3, level)]))
            ops = [self.rng.choice(self.prim_keys) for _ in range(n_comp)]

        ex1 = self.rng.sample(range(8), 3)
        ex2 = self.rng.sample(range(8), 3)
        # Ensure ex1 and ex2 disambiguate even/odd when conditional rule is used
        if "ev_sw" in ops:
            ex1[0] = (ex1[0] // 2) * 2
            if ex1[0] in ex1[1:]:
                ex1[0] = next(v for v in [0, 2, 4, 6] if v not in ex1[1:])
            ex2[0] = (ex2[0] // 2) * 2 + 1
            if ex2[0] in ex2[1:]:
                ex2[0] = next(v for v in [1, 3, 5, 7] if v not in ex2[1:])
        tst = self.rng.sample(range(8), 3)

        o1 = apply_composed_rule(ex1, ops)
        o2 = apply_composed_rule(ex2, ops)
        ot = apply_composed_rule(tst, ops)
        mid = apply_composed_rule(tst, ops[:1])

        fmt = lambda v: f"{v[0]},{v[1]},{v[2]}"
        rule_desc = " then ".join(PRIMITIVE_RULES[k][0] for k in ops)
        prompt = f"induce: [{fmt(ex1)}] -> [{fmt(o1)}] ; [{fmt(ex2)}] -> [{fmt(o2)}] ; [{fmt(tst)}] ->"
        think = f"rule {rule_desc} -> {fmt(mid)}"
        ans = fmt(ot)
        return ChallengeItem("induce", level, f"induce:{'+'.join(ops)}", prompt, think, ans)

    def _make_py(self, level: int, _parent_gene: str | None) -> ChallengeItem:
        a, b, c = self.rng.randint(0, 7), self.rng.randint(0, 7), self.rng.randint(0, 7)
        clk = self.rng.randint(0, 11)
        min_ops = 1 if level == 1 else (4 if level == 2 else 7 + 2 * (level - 3))
        max_ops = 2 if level == 1 else (6 if level == 2 else 9 + 2 * (level - 3))
        n_ops = self.rng.randint(min_ops, max_ops)

        # Level 1 teaches atomic 1-2 step primitives (including basic conditionals);
        # Level 2+ requires chaining 4-10+ multi-step state updates & fillers!
        pool = ["tick", "swap_ab", "swap_bc", "write", "filler", "if_clk_swap", "if_a_swap"]

        lines = [f"py: set a= {a} ; set b= {b} ; set c= {c} ; set clk= {clk}"]
        for _ in range(n_ops):
            op = self.rng.choice(pool)
            if op == "filler":
                lines.append(self.rng.choice(FILLERS))
            elif op == "tick":
                dh = self.rng.choice([1, 2, 5, 11])
                clk = (clk + dh) % 12
                lines.append(f"tick clk {'-1' if dh == 11 else f'+{dh}'}")
            elif op == "swap_ab":
                a, b = b, a
                lines.append("swap a,b")
            elif op == "swap_bc":
                b, c = c, b
                lines.append("swap b,c")
            elif op == "if_clk_swap":
                if clk % 2 == 0:
                    a, b = b, a
                lines.append("if clk even then swap a,b")
            elif op == "if_a_swap":
                if a % 2 == 0:
                    b, c = c, b
                lines.append("if a even then swap b,c")
            elif op == "write":
                vr = self.rng.choice(["a", "b", "c"])
                vl = self.rng.randint(0, 7)
                if vr == "a":
                    a = vl
                elif vr == "b":
                    b = vl
                else:
                    c = vl
                lines.append(f"set {vr}= {vl}")

        prompt = " ; ".join(lines) + " ->"
        par = "even" if clk % 2 == 0 else "odd"
        think = f"clk={clk} {par} a={a} b={b} c={c}"
        ans = f"{a},{b},{c},{clk}"
        return ChallengeItem("py", level, f"py:L{level}_ops{n_ops}", prompt, think, ans)

    def _make_story(self, level: int, _parent_gene: str | None) -> ChallengeItem:
        vals = [self.rng.randint(1, 7) for _ in range(3)]
        dial = self.rng.randint(0, 11)
        n_ev = self.rng.randint(1, 2) if level == 1 else (self.rng.randint(4, 6) if level == 2 else self.rng.randint(7, 9 + 2 * (level - 3)))

        pool = ["turn", "trade01", "trade12", "chatter", "if_ava", "if_dial"]

        clauses = [f"story: set Ava= {vals[0]} , set Ben= {vals[1]} , set Zoe= {vals[2]} , set dial= {dial}."]
        for _ in range(n_ev):
            ev = self.rng.choice(pool)
            if ev == "chatter":
                clauses.append(self.rng.choice(FILLERS))
            elif ev == "turn":
                st = self.rng.choice([1, 2, 5])
                dial = (dial + st) % 12
                clauses.append(f"tick dial +{st}.")
            elif ev == "trade01":
                vals[0], vals[1] = vals[1], vals[0]
                clauses.append("swap Ava,Ben.")
            elif ev == "trade12":
                vals[1], vals[2] = vals[2], vals[1]
                clauses.append("swap Ben,Zoe.")
            elif ev == "if_ava":
                if vals[0] % 2 == 0:
                    vals[1], vals[2] = vals[2], vals[1]
                clauses.append("if Ava even then swap Ben,Zoe.")
            elif ev == "if_dial":
                if dial % 2 == 0:
                    vals[0], vals[1] = vals[1], vals[0]
                clauses.append("if dial even then swap Ava,Ben.")

        prompt = " ".join(clauses) + " query ->"
        think = f"dial={dial} Ava={vals[0]} Ben={vals[1]} Zoe={vals[2]}"
        ans = f"{vals[0]},{vals[1]},{vals[2]},{dial}"
        return ChallengeItem("story", level, f"story:L{level}_ev{n_ev}", prompt, think, ans)
