"""
Philosopher 1.0 Tokenizer: Fast Persistent Two-Stage SuperBPE + Compositional CoBPE
===================================================================================
Trains in <2s over thousands of AI/RSI/logic `<think>` traces + algorithmic puzzles
using frequency-aggregated pretoken tables and fast cached `encode_span()`.
"""

import json
from collections import Counter


SURFACE_MODIFIERS = [
    "<NONE>",
    " ",
    "if ",
    "then ",
    "swap ",
    "set ",
    "tick ",
    "rule ",
    "= ",
    "-> ",
    "[L1] ",
    "[L2] ",
    "[SOLVE] ",
    "def ",
]


class PhilosopherTokenizer:
    def __init__(self, subword_merges: int = 720, superword_merges: int = 220):
        self.subword_merges = subword_merges
        self.superword_merges = superword_merges
        self.mod_to_id = {m: i for i, m in enumerate(SURFACE_MODIFIERS)}
        self.id_to_mod = {i: m for i, m in enumerate(SURFACE_MODIFIERS)}

        self.special_tokens = ["<PAD>", "<MASK>", "<BOS>", "<EOS>", "<think>", "</think>", "<ans>", "</ans>"]
        self.base_to_id: dict[str, int] = {}
        self.id_to_base: dict[int, str] = {}
        self.stage1_merges: list[tuple[str, str]] = []
        self.stage2_merges: list[tuple[str, str]] = []
        self._word_s1_cache: dict[str, list[str]] = {}

    def _split_pretokens_whitespace(self, text: str) -> list[str]:
        """
        Clean compositional pretokenization:
          - Keeps special tokens intact
          - Keeps multi-digit integers (`10`, `11`) as clean standalone number tokens
            (never glued to `.` or `,` or `]`), while preserving operator/variable words
            (`a=`, `a,b`, `Ben,Zoe`) so SuperBPE + CoBPE merges phrases cleanly!
        """
        words: list[str] = []
        i = 0
        n = len(text)
        bracket_punct = set("[];.():")
        while i < n:
            matched_sp = None
            for sp in self.special_tokens:
                if text.startswith(sp, i):
                    matched_sp = sp
                    break
            if matched_sp is not None:
                words.append(matched_sp)
                i += len(matched_sp)
                continue

            start = i
            if text[i] == " ":
                i += 1
                if i >= n:
                    words.append(" ")
                    break

            if any(text.startswith(sp, i) for sp in self.special_tokens):
                words.append(text[start:i])
                continue

            if text[i].isdigit():
                while i < n and text[i].isdigit():
                    i += 1
                words.append(text[start:i])
                continue

            if text[i] in bracket_punct or (text[i] == "," and (i + 1 >= n or text[i + 1].isdigit() or text[i + 1] == " ")):
                i += 1
                words.append(text[start:i])
                continue

            while (
                i < n
                and text[i] != " "
                and (not text[i].isdigit())
                and (text[i] not in bracket_punct)
                and not any(text.startswith(sp, i) for sp in self.special_tokens)
            ):
                if text[i] == "," and (i + 1 >= n or text[i + 1].isdigit() or text[i + 1] == " "):
                    break
                i += 1
            w = text[start:i]
            if w:
                words.append(w)
        return words

    @staticmethod
    def _apply_merge_to_seq(seq: list[str], pair: tuple[str, str]) -> list[str]:
        out: list[str] = []
        i = 0
        a, b = pair
        ab = a + b
        n = len(seq)
        while i < n:
            if i + 1 < n and seq[i] == a and seq[i + 1] == b:
                out.append(ab)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        return out

    def _decompose_cobpe(self, supertok: str) -> tuple[str, int]:
        if supertok in self.special_tokens:
            return supertok, 0
        best_mod = ""
        best_id = 0
        stripped = supertok.lstrip(" ")
        leading_sp = supertok[: len(supertok) - len(stripped)]

        for mod_str in SURFACE_MODIFIERS[2:]:
            if stripped.startswith(mod_str) and len(stripped) > len(mod_str):
                if len(mod_str) > len(best_mod):
                    best_mod = mod_str
                    best_id = self.mod_to_id[mod_str]

        if best_id != 0:
            return stripped[len(best_mod) :], best_id
        if leading_sp == " " and len(stripped) > 0:
            return stripped, self.mod_to_id[" "]
        return supertok, 0

    def train(self, texts: list[str]) -> dict[str, float]:
        special_set = set(self.special_tokens)
        raw_docs = [self._split_pretokens_whitespace(t) for t in texts]

        # Stage 1: Fast frequency-weighted subword BPE over unique pretokens!
        # Keep multi-digit integers (`10`, `11`, `896`) atomic so `"11"` never splits into `"1"+"1"`!
        word_freqs = Counter(w for doc in raw_docs for w in doc if w not in special_set)
        word_splits: dict[str, list[str]] = {}
        for w in word_freqs:
            st_w = w.lstrip(" ")
            if st_w.isdigit() and len(st_w) <= 4:
                word_splits[w] = [w]
            else:
                word_splits[w] = list(w)

        for _ in range(self.subword_merges):
            pair_counts: Counter[tuple[str, str]] = Counter()
            for w, freq in word_freqs.items():
                 pieces = word_splits[w]
                 for j in range(len(pieces) - 1):
                     pair_counts[(pieces[j], pieces[j + 1])] += freq
            if not pair_counts:
                break
            best_pair, best_freq = pair_counts.most_common(1)[0]
            if best_freq < 5:
                break
            self.stage1_merges.append(best_pair)
            a, b = best_pair
            for w, pieces in word_splits.items():
                if a in pieces and b in pieces:
                    word_splits[w] = self._apply_merge_to_seq(pieces, best_pair)

        self._word_s1_cache = dict(word_splits)
        bpe_len = sum(
            (1 if w in special_set else len(word_splits[w]))
            for doc in raw_docs
            for w in doc
        )

        # Stage 2: SuperBPE cross-word supertoken merges on evenly strided slice across ALL domains!
        # Never merge across digits or structural delimiters so numbers & clauses stay compositional!
        stride = max(1, len(raw_docs) // 900)
        sample_docs = raw_docs[::stride][:900]
        flat_docs = [
            [tok for w in doc for tok in ([w] if w in special_set else word_splits[w])]
            for doc in sample_docs
        ]
        no_super_chars = set("0123456789[];.(){}:")
        for _ in range(self.superword_merges):
            s_counts: Counter[tuple[str, str]] = Counter()
            for doc in flat_docs:
                for j in range(len(doc) - 1):
                    t1, t2 = doc[j], doc[j + 1]
                    if t1 in special_set or t2 in special_set:
                        continue
                    if any(c in no_super_chars for c in t1) or any(c in no_super_chars for c in t2):
                        continue
                    s_counts[(t1, t2)] += 1
            if not s_counts:
                break
            best_pair, freq = s_counts.most_common(1)[0]
            if freq < 6:
                break
            self.stage2_merges.append(best_pair)
            flat_docs = [self._apply_merge_to_seq(doc, best_pair) for doc in flat_docs]

        super_sample_len = sum(len(d) for d in flat_docs)
        bpe_sample_len = sum(
            (1 if w in special_set else len(word_splits[w]))
            for doc in sample_docs
            for w in doc
        )

        vocab_items = list(self.special_tokens)
        seen = set(vocab_items)
        for num_s in [str(k) for k in range(20)]:
            if num_s not in seen:
                vocab_items.append(num_s)
                seen.add(num_s)
        for ch in (
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            "_+-=*/()[]{}<>,.:;?!'\"\\|@#$%^&~`\n "
        ):
            if ch not in seen:
                vocab_items.append(ch)
                seen.add(ch)

        # Collect all Stage-1 & Stage-2 CoBPE base concepts
        for pieces in word_splits.values():
            for p in pieces:
                base, _ = self._decompose_cobpe(p)
                if base not in seen:
                    vocab_items.append(base)
                    seen.add(base)
        for doc in flat_docs:
            for supertok in doc:
                base, _ = self._decompose_cobpe(supertok)
                if base not in seen:
                    vocab_items.append(base)
                    seen.add(base)

        self.base_to_id = {b: idx for idx, b in enumerate(vocab_items)}
        self.id_to_base = {idx: b for idx, b in enumerate(vocab_items)}

        return {
            "compression_vs_bpe_pct": (1.0 - super_sample_len / max(1, bpe_sample_len)) * 100.0,
            "base_vocab_size": len(self.base_to_id),
            "modifier_vocab_size": len(self.mod_to_id),
            "total_bpe_tokens": bpe_len,
        }

    def _encode_word_stage1(self, w: str) -> list[str]:
        if w in self.special_tokens:
            return [w]
        cached = self._word_s1_cache.get(w)
        if cached is not None:
            return cached
        st_w = w.lstrip(" ")
        if st_w.isdigit() and len(st_w) <= 4:
            self._word_s1_cache[w] = [w]
            return [w]
        seq = list(w)
        for pair in self.stage1_merges:
            if pair[0] in seq and pair[1] in seq:
                seq = self._apply_merge_to_seq(seq, pair)
        if len(self._word_s1_cache) < 50000:
            self._word_s1_cache[w] = seq
        return seq

    def encode_span(self, text: str, add_bos: bool = False, add_eos: bool = False) -> tuple[list[int], list[int]]:
        words = self._split_pretokens_whitespace(text)
        flat = [tok for w in words for tok in self._encode_word_stage1(w)]
        if flat:
            tok_set = set(flat)
            for pair in self.stage2_merges:
                if pair[0] in tok_set and pair[1] in tok_set:
                    flat = self._apply_merge_to_seq(flat, pair)
                    tok_set.add(pair[0] + pair[1])

        base_ids: list[int] = [self.base_to_id["<BOS>"]] if add_bos else []
        mod_ids: list[int] = [0] if add_bos else []
        for tok in flat:
            base, mid = self._decompose_cobpe(tok)
            bid = self.base_to_id.get(base)
            if bid is not None:
                base_ids.append(bid)
                mod_ids.append(mid)
            else:
                if mid != 0:
                    m_str = self.id_to_mod[mid]
                    for ch in m_str:
                        base_ids.append(self.base_to_id.get(ch, 0))
                        mod_ids.append(0)
                for ch in base:
                    base_ids.append(self.base_to_id.get(ch, 0))
                    mod_ids.append(0)
        if add_eos:
            base_ids.append(self.base_to_id["<EOS>"])
            mod_ids.append(0)
        return base_ids, mod_ids

    def decode(self, base_ids: list[int], mod_ids: list[int], show_masks: bool = False) -> str:
        pieces = []
        for bid, mid in zip(base_ids, mod_ids):
            b_str = self.id_to_base.get(bid, "")
            if b_str in ("<PAD>", "<BOS>", "<EOS>"):
                continue
            if b_str == "<MASK>":
                if show_masks:
                    pieces.append("▪")
                continue
            m_str = "" if mid == 0 else self.id_to_mod.get(mid, "")
            if mid >= 2:
                pieces.append(" " + m_str + b_str)
            else:
                pieces.append(m_str + b_str)
        return "".join(pieces)

    def save(self, path: str):
        data = {
            "subword_merges": self.subword_merges,
            "superword_merges": self.superword_merges,
            "stage1_merges": self.stage1_merges,
            "stage2_merges": self.stage2_merges,
            "vocab_list": [self.id_to_base[i] for i in range(len(self.id_to_base))],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PhilosopherTokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(data["subword_merges"], data["superword_merges"])
        tok.stage1_merges = [tuple(p) for p in data["stage1_merges"]]
        tok.stage2_merges = [tuple(p) for p in data["stage2_merges"]]
        tok.base_to_id = {b: i for i, b in enumerate(data["vocab_list"])}
        tok.id_to_base = {i: b for i, b in enumerate(data["vocab_list"])}
        return tok
