"""
Philosopher 1.0 Brain — 40-Layer Core + 4 Extra Task-Specialist Layers (44-Layer Hybrid)
========================================================================================
Cost-effective ultra-deep reasoning & invention stack powered by TRM weight-sharing,
DeepSeek-V4.1-Flash Asymmetric CED + mHC + CSA2 + Engram, Kimi K3.1 Wreath-KDA +
Gated MLA + AttnRes + Stable LatentMoE, and Dual-Gear NC^1 Wreath State Tracking:

  1. 20-Layer Causal Encoder (Layers 1..20):
       - Layers 1..19 : 3 Weight-Shared `DeepSemanticKDABlock` groups looped (6+6+7 = 19 layers)
                        with per-layer `enc_depth_emb` & `1/19` mHC manifold scaling (~114-tok horizon!).
       - Layer 20     : Dual-Gear `Wreath-KDA` (`8x8` {-1,0,+1} Swap-Boxes + `4x` 30° Clock-Dials).
  2. 20-Layer Recursive NCM + TRM Block-Diffusion Decoder (Layers 21..40):
       - 4 Coarse-to-Fine Denoising Passes x 5 Sub-Layers/Pass = 20 Effective Decoder Layers:
           * Sub-Layer 1 : Bidirectional Self-Attn + `CSA2`-Reused `Gated-MLA` Cross-Attn + `AttnRes`
           * Sub-Layer 2 : Kimi K3.1 `Stable LatentMoE` (1 Shared + Top-2 of 4 Latent Experts)
           * Sub-Layers 3..5 : 3 Recursive `TRM` Deduction Micro-Layers with `dec_depth_emb[0..19]` & mHC!
  3. +4 Extra Task-Specialized Layers (Layers 41..44):
       - Dedicated specialist blocks (`S0: Rule-Invention`, `S1: Code-State`, `S2: Logic-SAT`,
         `S3: AI-RSI-Theory`) dynamically routed from `p_state` (Top-2 active per prompt)!
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


THINK_LEN = 28
ANS_LEN = 12
BLOCK_LEN = THINK_LEN + ANS_LEN  # 40 parallel supertokens
ENGRAM_BUCKETS = 512
SPECIALIST_NAMES = ["Invention", "CodeState", "LogicSAT", "AITheory"]


def apply_deadzone_gate(raw_beta: torch.Tensor, zone: float = 0.05, active: bool = True) -> torch.Tensor:
    b = torch.sigmoid(raw_beta)
    if not active or zone <= 0.0:
        return b
    return torch.clamp((b - zone) / (1.0 - 2.0 * zone), 0.0, 1.0)


def snap_signed_perm(U: torch.Tensor) -> torch.Tensor:
    B, d, _ = U.shape
    out = torch.zeros_like(U)
    work = U.abs().clone()
    signs = U.sign()
    ar = torch.arange(B, device=U.device)
    for _ in range(d):
        idx = work.view(B, -1).argmax(dim=-1)
        r, c = idx // d, idx % d
        out[ar, r, c] = signs[ar, r, c]
        work[ar, r, :] = -1.0
        work[ar, :, c] = -1.0
    return out


def snap_clock_30deg(U_clk: torch.Tensor) -> torch.Tensor:
    sin_e = 0.5 * (U_clk[:, 1, 0] - U_clk[:, 0, 1])
    cos_e = 0.5 * (U_clk[:, 0, 0] + U_clk[:, 1, 1])
    th = torch.atan2(sin_e.float(), cos_e.float())
    step = 2.0 * math.pi / 12.0
    th_s = torch.round(th / step) * step
    c, s = torch.cos(th_s).to(U_clk.dtype), torch.sin(th_s).to(U_clk.dtype)
    return torch.stack([torch.stack([c, -s], dim=-1), torch.stack([s, c], dim=-1)], dim=-2)


def _prefix_matmul_rtl(M: torch.Tensor) -> torch.Tensor:
    T = M.shape[-3]
    P = M
    st = 1
    while st < T:
        merged = torch.matmul(P[..., st:, :, :], P[..., :-st, :, :])
        P = torch.cat([P[..., :st, :, :], merged], dim=-3)
        st *= 2
    return P


def gpu_chunk_scan_swap(G: torch.Tensor, chunk_c: int = 4, snap: bool = False) -> torch.Tensor:
    B, L, d, _ = G.shape
    pad = (chunk_c - (L % chunk_c)) % chunk_c
    if pad > 0:
        eye_p = torch.eye(d, device=G.device, dtype=G.dtype).view(1, 1, d, d).expand(B, pad, d, d)
        Gp = torch.cat([G, eye_p], dim=1)
    else:
        Gp = G
    N_c = Gp.shape[1] // chunk_c
    U_intra = _prefix_matmul_rtl(Gp.view(B, N_c, chunk_c, d, d))
    totals = U_intra[:, :, -1]
    if snap:
        t_snap = snap_signed_perm(totals.reshape(-1, d, d)).view(B, N_c, d, d)
        totals = totals + (t_snap - totals).detach()
    if N_c > 1:
        U_inter = _prefix_matmul_rtl(totals)
        if snap:
            ui_snap = snap_signed_perm(U_inter.reshape(-1, d, d)).view(B, N_c, d, d)
            U_inter = U_inter + (ui_snap - U_inter).detach()
        eye0 = torch.eye(d, device=G.device, dtype=G.dtype).view(1, 1, d, d).expand(B, 1, d, d)
        prev = torch.cat([eye0, U_inter[:, :-1]], dim=1)
        U_full = torch.matmul(U_intra, prev.unsqueeze(2))
    else:
        U_full = U_intra
    return U_full.reshape(B, -1, d, d)[:, :L]


def gpu_chunk_scan_clock(R: torch.Tensor, chunk_c: int = 4, snap: bool = False) -> torch.Tensor:
    B, L, D, _, _ = R.shape
    Rf = R.permute(0, 2, 1, 3, 4).reshape(B * D, L, 2, 2)
    pad = (chunk_c - (L % chunk_c)) % chunk_c
    if pad > 0:
        eye_p = torch.eye(2, device=R.device, dtype=R.dtype).view(1, 1, 2, 2).expand(B * D, pad, 2, 2)
        Rp = torch.cat([Rf, eye_p], dim=1)
    else:
        Rp = Rf
    N_c = Rp.shape[1] // chunk_c
    U_intra = _prefix_matmul_rtl(Rp.view(B * D, N_c, chunk_c, 2, 2))
    totals = U_intra[:, :, -1]
    if snap:
        t_snap = snap_clock_30deg(totals.reshape(-1, 2, 2)).view(B * D, N_c, 2, 2)
        totals = totals + (t_snap - totals).detach()
    if N_c > 1:
        U_inter = _prefix_matmul_rtl(totals)
        if snap:
            ui_snap = snap_clock_30deg(U_inter.reshape(-1, 2, 2)).view(B * D, N_c, 2, 2)
            U_inter = U_inter + (ui_snap - U_inter).detach()
        eye0 = torch.eye(2, device=R.device, dtype=R.dtype).view(1, 1, 2, 2).expand(B * D, 1, 2, 2)
        prev = torch.cat([eye0, U_inter[:, :-1]], dim=1)
        U_full = torch.matmul(U_intra, prev.unsqueeze(2))
    else:
        U_full = U_intra
    return U_full.reshape(B, D, -1, 2, 2)[:, :, :L].permute(0, 2, 1, 3, 4)


def gpu_chunk_scan_delta(k_eff: torch.Tensor, v: torch.Tensor, beta: torch.Tensor, chunk_c: int = 16) -> torch.Tensor:
    B, L, d_s = k_eff.shape
    d_v = v.shape[-1]
    pad = (chunk_c - (L % chunk_c)) % chunk_c
    if pad > 0:
        k_eff = F.pad(k_eff, (0, 0, 0, pad))
        v = F.pad(v, (0, 0, 0, pad))
        beta = F.pad(beta, (0, 0, 0, pad))
    Lp = k_eff.shape[1]
    N_c = Lp // chunk_c
    ke = k_eff.view(B, N_c, chunk_c, d_s)
    vc = v.view(B, N_c, chunk_c, d_v)
    bc = beta.view(B, N_c, chunk_c, 1)

    Id = torch.eye(d_s, device=k_eff.device, dtype=k_eff.dtype).view(1, 1, 1, d_s, d_s)
    A = Id - bc.unsqueeze(-1) * (ke.unsqueeze(-1) * ke.unsqueeze(-2))
    B_op = (vc * bc).unsqueeze(-1) * ke.unsqueeze(-2)

    st = 1
    while st < chunk_c:
        Al, Bl = A[:, :, :-st], B_op[:, :, :-st]
        Ar, Br = A[:, :, st:], B_op[:, :, st:]
        A = torch.cat([A[:, :, :st], torch.matmul(Al, Ar)], dim=2)
        B_op = torch.cat([B_op[:, :, :st], torch.matmul(Bl, Ar) + Br], dim=2)
        st *= 2

    if N_c > 1:
        Ac, Bc = A[:, :, -1], B_op[:, :, -1]
        st = 1
        while st < N_c:
            Al, Bl = Ac[:, :-st], Bc[:, :-st]
            Ar, Br = Ac[:, st:], Bc[:, st:]
            Ac = torch.cat([Ac[:, :st], torch.matmul(Al, Ar)], dim=1)
            Bc = torch.cat([Bc[:, :st], torch.matmul(Bl, Ar) + Br], dim=1)
            st *= 2
        z0 = torch.zeros(B, 1, d_v, d_s, device=k_eff.device, dtype=k_eff.dtype)
        S_prev = torch.cat([z0, Bc[:, :-1]], dim=1)
        S_full = torch.matmul(S_prev.unsqueeze(2), A) + B_op
    else:
        S_full = B_op
    return S_full.reshape(B, Lp, d_v, d_s)[:, :L]


def gpu_scan_clock_mem(c_beta: torch.Tensor, w_hand: torch.Tensor) -> torch.Tensor:
    a = (1.0 - c_beta).unsqueeze(-1)
    b = c_beta.unsqueeze(-1) * w_hand
    st = 1
    L = c_beta.shape[1]
    while st < L:
        al, bl = a[:, :-st], b[:, :-st]
        ar, br = a[:, st:], b[:, st:]
        a = torch.cat([a[:, :st], ar * al], dim=1)
        b = torch.cat([b[:, :st], ar * bl + br], dim=1)
        st *= 2
    return b


class RecursiveEncoderKDAGroup(nn.Module):
    """
    Weight-shared Kimi K3.1 Deep Concept Refiner used recursively across Encoder Layers 1..19!
    Refines token-level AI/math/logic abstractions across 19 layers without creating an
    un-swapped wide-conv shortcut around Layer 20's Wreath permutation gears!
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.in_uv = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        nn.init.normal_(self.out_proj.weight, std=0.008)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, depth_vec: torch.Tensor, mhc_scale: torch.Tensor, not_pad: torch.Tensor) -> torch.Tensor:
        h = self.norm(x + depth_vec)
        u, gate = self.in_uv(h).chunk(2, dim=-1)
        delta = self.out_proj(F.silu(u) * torch.sigmoid(gate))
        return (x + mhc_scale * delta) * not_pad


class RecursiveDecoderTRMMicroLayer(nn.Module):
    """
    Fast weight-shared Pointwise SwiGLU TRM Deduction Micro-Layer (Sub-Layers 3..5 per diffusion pass,
    providing 12 of the 20 Decoder Layers with ZERO horizontal conv smearing across adjacent digits!).
    """

    def __init__(self, d_model: int = 128):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.in_ug = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        nn.init.normal_(self.out_proj.weight, std=0.005)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z: torch.Tensor, depth_vec: torch.Tensor, mhc_scale: torch.Tensor) -> torch.Tensor:
        h = self.norm(z + depth_vec)
        u, gate = self.in_ug(h).chunk(2, dim=-1)
        return z + mhc_scale * self.out_proj(F.silu(gate) * u)


class StableLatentMoE(nn.Module):
    """
    Kimi K3.1 Stable LatentMoE (Zero-Gather Single-GEMM Implementation):
      1 Shared Expert + Top-2 Masked Latent Experts using pure native `nn.Linear` GEMMs!
    """

    def __init__(self, d_model: int = 128, n_experts: int = 4, top_k: int = 2, d_lat: int = 64):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.d_lat = d_lat
        self.norm = nn.LayerNorm(d_model)
        self.shared = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.exp_up = nn.Linear(d_model, n_experts * d_lat)
        self.exp_down = nn.Linear(n_experts * d_lat, d_model, bias=False)
        nn.init.normal_(self.exp_down.weight, std=0.02)
        self.last_loads = torch.zeros(n_experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        h = self.norm(x)
        sh_out = self.shared(h)

        r_logits = self.router(F.normalize(h, dim=-1)) * 3.0
        probs = F.softmax(r_logits, dim=-1)
        top_w, top_idx = probs.topk(self.top_k, dim=-1)
        top_w = top_w / top_w.sum(dim=-1, keepdim=True).clamp(min=1e-6)

        mask = torch.zeros_like(probs).scatter_(-1, top_idx, top_w)
        with torch.no_grad():
            self.last_loads = (mask > 0).float().mean(dim=(0, 1)).detach().cpu()

        lat = F.silu(self.exp_up(h)).view(B, T, self.n_experts, self.d_lat)
        lat_gated = (lat * mask.unsqueeze(-1)).reshape(B, T, self.n_experts * self.d_lat)
        return sh_out + self.exp_down(lat_gated)


class TaskSpecialistBank(nn.Module):
    """
    +4 Extra Task-Specialized Layers (Layers 41..44, Pointwise Position-Sharp SwiGLU):
      - Specialist 0: Rule-Invention & Algebra (`induce:`, mystery operators)
      - Specialist 1: Stateful Code & Execution (`py:`, debugging)
      - Specialist 2: Deductive Logic & Stories (`story:`, 3-SAT, CSP)
      - Specialist 3: AI Architecture & RSI Theory (`think:`, V4.1-Flash, K3.1, SVF)
    Dynamically routed per-prompt from `p_state` (Top-2 active)!
    """

    def __init__(self, d_model: int = 128, n_spec: int = 4, d_hid: int = 96):
        super().__init__()
        self.n_spec = n_spec
        self.d_hid = d_hid
        self.norm = nn.LayerNorm(d_model)
        self.task_router = nn.Linear(d_model, n_spec)
        self.spec_up = nn.Linear(d_model, n_spec * d_hid)
        self.spec_down = nn.Linear(n_spec * d_hid, d_model, bias=False)
        nn.init.normal_(self.spec_down.weight, std=0.005)
        self.last_spec_weights = torch.zeros(n_spec)

    def forward(self, z: torch.Tensor, p_state: torch.Tensor) -> torch.Tensor:
        B, T, _ = z.shape
        r_logits = self.task_router(F.normalize(p_state.detach(), dim=-1)) * 3.0
        probs = F.softmax(r_logits, dim=-1)
        top_w, top_idx = probs.topk(2, dim=-1)
        top_w = top_w / top_w.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        mask = torch.zeros_like(probs).scatter_(-1, top_idx, top_w)

        with torch.no_grad():
            self.last_spec_weights = mask.mean(dim=0).detach().cpu()

        h = self.norm(z)
        act = F.silu(self.spec_up(h)).view(B, T, self.n_spec, self.d_hid)
        gated = (act * mask.view(B, 1, self.n_spec, 1)).reshape(B, T, self.n_spec * self.d_hid)
        return z + self.spec_down(gated)


class PhilosopherBrain(nn.Module):
    """
    40-Layer Core (20 Encoder + 20 Decoder) + 4 Task-Specialist Extra Layers = 44 Layers!
    """

    def __init__(
        self,
        base_vocab: int,
        mod_vocab: int,
        mask_base_id: int,
        d_model: int = 128,
        n_heads: int = 8,
        d_swap: int = 8,
        d_v: int = 32,
        n_dials: int = 4,
        chunk_c: int = 4,
        block_len: int = BLOCK_LEN,
    ):
        super().__init__()
        self.base_vocab = base_vocab
        self.mod_vocab = mod_vocab
        self.mask_base_id = mask_base_id
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_swap = d_swap
        self.d_v = d_v
        self.n_dials = n_dials
        self.chunk_c = chunk_c
        self.block_len = block_len
        self.core_layers = 40
        self.specialist_layers = 4
        self.total_effective_layers = 44

        self.base_emb = nn.Embedding(base_vocab, d_model)
        self.mod_emb = nn.Embedding(mod_vocab, d_model)
        self.engram_table = nn.Embedding(ENGRAM_BUCKETS, d_model)
        nn.init.normal_(self.engram_table.weight, std=0.01)
        self.engram_gate = nn.Parameter(torch.tensor(0.05))

        # Clean local 4-token phrase conv (preserves exact variable-value binding!)
        self.prompt_conv = nn.Conv1d(d_model, d_model, kernel_size=4, padding=3, groups=d_model)
        self.prompt_conv_proj = nn.Linear(d_model, d_model)

        # ---- 20-LAYER ASYMMETRIC CAUSAL ENCODER (Layers 1..19 Recursive KDA + Layer 20 Wreath-KDA) ----
        self.enc_groups = nn.ModuleList([RecursiveEncoderKDAGroup(d_model) for _ in range(3)])
        self.enc_schedule = [0] * 6 + [1] * 6 + [2] * 7  # 6 + 6 + 7 = 19 Semantic Layers!
        self.enc_depth_emb = nn.Parameter(torch.randn(19, 1, 1, d_model) * 0.01)
        self.enc_mhc_scales = nn.Parameter(torch.full((19,), 0.25))  # Scaled by 1/sqrt(19) in forward

        # Layer 20: Dual-Gear Wreath-KDA + 4x 30° Clock-Dials
        self.w2_norm = nn.LayerNorm(d_model)
        self.to_q_heads = nn.Linear(d_model, n_heads * d_swap)
        self.to_k = nn.Linear(d_model, d_swap)
        self.to_v = nn.Linear(d_model, d_v)
        self.to_beta = nn.Linear(d_model, 1)
        self.kda_out_gate = nn.Linear(d_model, n_heads * d_v)
        nn.init.zeros_(self.kda_out_gate.weight)
        nn.init.zeros_(self.kda_out_gate.bias)

        self.w1_norm = nn.LayerNorm(d_model)
        self.to_clk_theta = nn.Linear(d_model, n_dials)
        self.to_clk_phi = nn.Linear(d_model, n_dials)
        self.to_clk_beta = nn.Linear(d_model, n_dials)

        ctx_in_dim = d_model + 2 * n_dials + d_v
        self.w1_cascade_fusion = nn.Sequential(
            nn.LayerNorm(ctx_in_dim),
            nn.Linear(ctx_in_dim, d_model),
            nn.SiLU(),
        )
        self.to_hh_k = nn.Linear(d_model, 2 * d_swap)
        self.to_hh_b = nn.Linear(d_model, 2)

        with torch.no_grad():
            self.to_beta.bias.fill_(-2.2)
            self.to_hh_b.bias.fill_(-2.5)
            self.to_clk_beta.bias.fill_(-2.2)
            self.to_clk_theta.weight.mul_(0.08)
            self.to_clk_theta.bias.zero_()

        self.box_slot_proj = nn.Linear(d_v, d_model)
        self.clk_slot_proj = nn.Linear(2, d_model)
        self.slot_type_emb = nn.Embedding(24, d_model)
        self.sem_pool_proj = nn.Linear(d_model, d_model)
        self.prose_state_proj = nn.Sequential(
            nn.LayerNorm(2 * d_model),
            nn.Linear(2 * d_model, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.mem_kv = nn.Linear(d_model, 2 * d_model)
        self.mla_gate = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.mla_gate.weight)
        nn.init.zeros_(self.mla_gate.bias)

        summary_dim = n_heads * d_v + 2 * n_dials + 2 * d_model
        self.prompt_state_proj = nn.Sequential(
            nn.LayerNorm(summary_dim),
            nn.Linear(summary_dim, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )

        # ---- 20-LAYER RECURSIVE NCM + TRM DECODER (4 Passes x 5 Sub-Layers = Layers 21..40) ----
        self.block_pos_emb = nn.Parameter(torch.randn(1, block_len, d_model) * 0.02)
        self.pass_emb = nn.Embedding(4, d_model)

        n_concepts = block_len // chunk_c
        self.ncm_planner = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_concepts * d_model),
        )

        init_ar = torch.full((4, 4), -2.0)
        for p_i in range(4):
            init_ar[p_i, p_i] = 3.5
        self.attn_res_logits = nn.Parameter(init_ar)
        self.mhc_mix_logit = nn.Parameter(torch.full((4,), -2.0))

        self.trm_norm1 = nn.LayerNorm(2 * d_model)
        self.trm_in_proj = nn.Linear(2 * d_model, d_model)
        self.block_self_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.prompt_cross_attn = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.trm_moe = StableLatentMoE(d_model=d_model, n_experts=4, top_k=2, d_lat=64)

        # 3 Recursive TRM Micro-Layers per pass (4 passes x 3 = 12 micro-layers + 4 Attn + 4 MoE = 20 Decoder Layers!)
        self.dec_micro = RecursiveDecoderTRMMicroLayer(d_model=d_model)
        self.dec_micro_depth_emb = nn.Parameter(torch.randn(12, 1, 1, d_model) * 0.01)
        self.dec_micro_mhc = nn.Parameter(torch.full((12,), 0.25))

        # ---- +4 EXTRA TASK-SPECIALIST LAYERS (Layers 41..44) ----
        self.specialist_bank = TaskSpecialistBank(d_model=d_model, n_spec=4, d_hid=96)

        self.out_norm = nn.LayerNorm(d_model)
        self.base_head = nn.Linear(d_model, base_vocab)
        self.mod_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, mod_vocab),
        )

    def encode_prompt(
        self,
        p_base: torch.Tensor,
        p_mod: torch.Tensor,
        clk_dz: float = 0.0,
        box_dz: float = 0.0,
        clk_noise: float = 0.0,
        box_noise: float = 0.0,
        snap_swap: bool = False,
        snap_clk: bool = False,
        prose_mode: bool = False,
    ):
        B, Lp = p_base.shape
        not_pad = (p_base != 0).float().unsqueeze(-1)
        e_mod = self.mod_emb(p_mod) * not_pad

        cls_b = torch.where((p_base >= 8) & (p_base <= 27), torch.full_like(p_base, 8), p_base)
        prev_cls = F.pad(cls_b[:, :-1], (1, 0), value=0)
        eng_idx = (prev_cls * 37 + p_mod * 13 + (cls_b % 16)) % ENGRAM_BUCKETS
        e_eng = self.engram_table(eng_idx) * (0.15 * torch.tanh(self.engram_gate)) * not_pad

        raw = (self.base_emb(p_base) + e_mod + e_eng) * not_pad
        conv = F.silu(self.prompt_conv(raw.transpose(1, 2))[:, :, :Lp].transpose(1, 2)) * not_pad
        h_local = (raw + self.prompt_conv_proj(conv)) * not_pad

        # ---- ENCODER LAYERS 1..19: 19 Recursive KDA Semantic Layers ----
        h_deep = h_local
        inv_s = 1.0 / math.sqrt(19.0)
        for l_idx, grp_id in enumerate(self.enc_schedule):
            m_scale = torch.tanh(self.enc_mhc_scales[l_idx]) * inv_s
            h_deep = self.enc_groups[grp_id](h_deep, self.enc_depth_emb[l_idx], m_scale, not_pad)

        if prose_mode:
            # Open-ended AI/RSI/SAT prose path: pool 19-layer `h_deep` into 16 semantic slots,
            # keeping Layer 20's algebraic 8x8 Wreath Swap-Boxes 100% protected from prose noise!
            n_sem_slots = min(16, Lp)
            stride = max(1, Lp // n_sem_slots)
            pooled = h_deep[:, -n_sem_slots * stride :: stride][:, :n_sem_slots]
            mem_slots = self.sem_pool_proj(pooled)
            slot_ids = torch.arange(mem_slots.shape[1], device=h_local.device).unsqueeze(0)
            mem_slots = mem_slots + self.slot_type_emb(slot_ids)

            denom_p = not_pad.sum(dim=1).clamp(min=1.0)
            mean_deep = (h_deep * not_pad).sum(dim=1) / denom_p
            p_state = self.prose_state_proj(torch.cat([mean_deep, h_deep[:, -1]], dim=-1))
            z_zero = torch.tensor(0.0, device=h_local.device)
            self.last_U_swap = torch.eye(self.d_swap, device=h_local.device).unsqueeze(0).expand(B, -1, -1)
            self.last_clk_final = torch.zeros(B, self.n_dials, 2, device=h_local.device)
            self.last_b2 = torch.zeros(B, Lp, 1, device=h_local.device)
            self.last_not_pad = not_pad.detach()
            return mem_slots, p_state, z_zero, z_zero

        # ---- ENCODER LAYER 20: Dual-Gear Wreath-KDA + 30° Clock-Dials (Algebraic State Path) ----
        w2 = self.w2_norm(h_local)
        q_heads = F.normalize(self.to_q_heads(w2).view(B, Lp, self.n_heads, self.d_swap), dim=-1)
        k = F.normalize(self.to_k(w2), dim=-1)
        if snap_swap:
            k_t = torch.where(k.abs() < 0.20, torch.zeros_like(k), k)
            k = k_t / k_t.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        v = self.to_v(w2)
        b_beta = apply_deadzone_gate(self.to_beta(w2), zone=box_dz, active=(box_dz > 0.0)) * not_pad

        w1_base = self.w1_norm(h_local + 0.5 * e_mod)
        rth = self.to_clk_theta(w1_base)
        if self.training and clk_noise > 0.0:
            rth = rth + torch.randn_like(rth) * (clk_noise * 0.5)

        theta = (math.pi * torch.tanh(rth)) * not_pad
        phi = math.pi * torch.tanh(self.to_clk_phi(w2))
        c_beta = apply_deadzone_gate(self.to_clk_beta(w2), zone=clk_dz, active=(clk_dz > 0.0)) * not_pad
        clk_mag = (0.5 * (1.0 - torch.cos(12.0 * theta))).mean() + (
            c_beta.detach() * 0.5 * (1.0 - torch.cos(12.0 * phi))
        ).mean()

        ct, st = torch.cos(theta), torch.sin(theta)
        R_clk = torch.stack([torch.stack([ct, -st], dim=-1), torch.stack([st, ct], dim=-1)], dim=-2)
        U_clk = gpu_chunk_scan_clock(R_clk, chunk_c=self.chunk_c, snap=snap_clk)

        reset_hand = torch.stack([torch.cos(phi), torch.sin(phi)], dim=-1)
        w_hand = torch.matmul(U_clk.transpose(-1, -2), reset_hand.unsqueeze(-1)).squeeze(-1)
        S_clk_all = gpu_scan_clock_mem(c_beta, w_hand)
        clk_seq = torch.matmul(U_clk, S_clk_all.unsqueeze(-1)).squeeze(-1).reshape(B, Lp, -1)

        S_skel = gpu_chunk_scan_delta(k, v, b_beta, chunk_c=16)
        box_skel_seq = torch.matmul(S_skel, q_heads[:, :, 0].unsqueeze(-1)).squeeze(-1)

        zpad_clk = torch.zeros(B, 1, 2 * self.n_dials, device=h_local.device)
        zpad_box = torch.zeros(B, 1, self.d_v, device=h_local.device)
        prev_ctx = torch.cat(
            [w1_base, torch.cat([zpad_clk, clk_seq[:, :-1]], dim=1), torch.cat([zpad_box, box_skel_seq[:, :-1]], dim=1)],
            dim=-1,
        )

        w1 = self.w1_cascade_fusion(prev_ctx) * not_pad
        rk = self.to_hh_k(w1).view(B, Lp, 2, self.d_swap)
        rb = self.to_hh_b(w1)
        if self.training and box_noise > 0.0:
            rk = rk + torch.randn_like(rk) * box_noise
            rb = rb + torch.randn_like(rb) * box_noise

        hk = F.normalize(rk, dim=-1)
        hb = (2.0 * torch.sigmoid(rb)) * not_pad
        Id = torch.eye(self.d_swap, device=h_local.device).view(1, 1, self.d_swap, self.d_swap)
        H1 = Id - hb[:, :, 0:1].unsqueeze(-1) * (hk[:, :, 0].unsqueeze(-1) * hk[:, :, 0].unsqueeze(-2))
        H2 = Id - hb[:, :, 1:2].unsqueeze(-1) * (hk[:, :, 1].unsqueeze(-1) * hk[:, :, 1].unsqueeze(-2))
        G_swap = torch.matmul(H2, H1)

        ortho_v = (((G_swap ** 2).sum(dim=-1) - 1.0) ** 2).mean()
        pm1_v = (self.d_swap - (G_swap ** 4).sum(dim=(-2, -1))).mean() / self.d_swap
        addr_v = (b_beta.detach().squeeze(-1) * (1.0 - (k ** 4).sum(dim=-1))).mean()
        box_mag = ortho_v + pm1_v + addr_v

        U_swap = gpu_chunk_scan_swap(G_swap, chunk_c=self.chunk_c, snap=snap_swap)
        k_eff = torch.matmul(U_swap.transpose(-1, -2), k.unsqueeze(-1)).squeeze(-1)
        S_all = gpu_chunk_scan_delta(k_eff, v, b_beta, chunk_c=16)

        q_eff_T = torch.bmm(U_swap[:, -1].transpose(-1, -2), q_heads[:, -1].transpose(-1, -2))
        raw_heads = torch.bmm(S_all[:, -1], q_eff_T).transpose(1, 2)
        kda_scale = 1.0 + 0.25 * torch.tanh(self.kda_out_gate(w2[:, -1])).view(B, self.n_heads, self.d_v)
        box_heads = raw_heads * kda_scale
        clk_final = clk_seq[:, -1].view(B, self.n_dials, 2)

        self.last_G_swap = G_swap.detach()
        self.last_theta = theta.detach()
        self.last_U_swap = U_swap[:, -1].detach()
        self.last_clk_final = clk_final.detach()
        self.last_b2 = b_beta.detach()
        self.last_not_pad = not_pad.detach()

        mem_slots = torch.cat(
            [self.box_slot_proj(box_heads), self.clk_slot_proj(clk_final), h_local[:, -1:], h_deep[:, -1:]], dim=1
        )  # [B, 14, d_model]
        slot_ids = torch.arange(mem_slots.shape[1], device=h_local.device).unsqueeze(0)
        mem_slots = mem_slots + self.slot_type_emb(slot_ids)

        summary = torch.cat([box_heads.reshape(B, -1), clk_final.reshape(B, -1), h_local[:, -1], h_deep[:, -1]], dim=-1)
        p_state = self.prompt_state_proj(summary)
        return mem_slots, p_state, box_mag, clk_mag

    def _denoise_pass(
        self,
        y_emb: torch.Tensor,
        z_history: list[torch.Tensor],
        mk: torch.Tensor,
        mv: torch.Tensor,
        p_state: torch.Tensor,
        pass_idx: int,
    ):
        n_hist = len(z_history)
        ar_weights = F.softmax(self.attn_res_logits[pass_idx, :n_hist], dim=0)
        z_res = sum(w_i * z_i for w_i, z_i in zip(ar_weights, z_history))

        # Sub-Layer 1 (Self-Attn + CSA2 Reused Gated-MLA Cross-Attn)
        x = (
            self.trm_in_proj(self.trm_norm1(torch.cat([y_emb, z_res], dim=-1)))
            + self.block_pos_emb
            + self.pass_emb.weight[pass_idx].view(1, 1, -1)
        )
        attn_self, _ = self.block_self_attn(x, x, x, need_weights=False)
        x = x + attn_self
        attn_cross, _ = self.prompt_cross_attn(x, mk, mv, need_weights=False)
        x = x + (1.0 + 0.20 * torch.tanh(self.mla_gate(x))) * attn_cross

        # Sub-Layer 2 (Stable LatentMoE + mHC Mixing)
        moe_delta = self.trm_moe(x)
        mhc_alpha = 0.15 * torch.sigmoid(self.mhc_mix_logit[pass_idx])
        z_cur = (1.0 - mhc_alpha) * z_res + mhc_alpha * x + moe_delta

        # Sub-Layers 3..5 (3 Recursive TRM Deduction Micro-Layers -> 5 Sub-Layers x 4 Passes = 20 Decoder Layers!)
        inv_d = 1.0 / math.sqrt(12.0)
        base_m = pass_idx * 3
        for m_i in range(3):
            idx_m = base_m + m_i
            d_scale = torch.tanh(self.dec_micro_mhc[idx_m]) * inv_d
            z_cur = self.dec_micro(z_cur, self.dec_micro_depth_emb[idx_m], d_scale)

        # +4 Extra Task-Specialist Layers (Layers 41..44, applied at top of 40-layer core on final pass!)
        if pass_idx == 3:
            z_next = self.specialist_bank(z_cur, p_state)
        else:
            z_next = z_cur

        h_out = self.out_norm(z_next)
        b_logits = self.base_head(h_out)
        pred_base_e = torch.matmul(F.softmax(b_logits, dim=-1), self.base_emb.weight)
        m_logits = self.mod_head(torch.cat([h_out, pred_base_e], dim=-1))
        return z_next, b_logits, m_logits

    def forward(
        self,
        p_base: torch.Tensor,
        p_mod: torch.Tensor,
        clk_dz: float = 0.0,
        box_dz: float = 0.0,
        clk_noise: float = 0.0,
        box_noise: float = 0.0,
        snap_swap: bool = False,
        snap_clk: bool = False,
        explore_temp: float = 0.0,
        prose_mode: bool = False,
    ):
        B = p_base.shape[0]
        mem_slots, p_state, b_mag, c_mag = self.encode_prompt(
            p_base,
            p_mod,
            clk_dz=clk_dz,
            box_dz=box_dz,
            clk_noise=clk_noise,
            box_noise=box_noise,
            snap_swap=snap_swap,
            snap_clk=snap_clk,
            prose_mode=prose_mode,
        )

        mk, mv = self.mem_kv(mem_slots).chunk(2, dim=-1)

        n_c = self.block_len // self.chunk_c
        coarse = self.ncm_planner(p_state).view(B, n_c, 1, self.d_model)
        z0 = coarse.expand(B, n_c, self.chunk_c, self.d_model).reshape(B, self.block_len, self.d_model)
        if explore_temp > 0.0:
            z0 = z0 + torch.randn_like(z0) * (0.16 * explore_temp)

        mask_emb = self.base_emb.weight[self.mask_base_id].view(1, 1, -1).expand(B, self.block_len, -1)
        z_history = [z0]
        z1, bl0, ml0 = self._denoise_pass(mask_emb, z_history, mk, mv, p_state, pass_idx=0)
        z_history.append(z1)
        b_logs = [bl0]
        m_logs = [ml0]

        for k_pass in range(1, 4):
            bl_prev = b_logs[-1]
            ml_prev = m_logs[-1]
            if explore_temp > 0.0 and k_pass in (1, 2):
                scale = (0.30 if k_pass == 1 else 0.15) * explore_temp
                gumbel = -torch.empty_like(bl_prev).exponential_().log() * scale
                probs_b = F.softmax(bl_prev + gumbel, dim=-1)
            else:
                probs_b = F.softmax(bl_prev, dim=-1)
            probs_m = F.softmax(ml_prev, dim=-1)

            conf_b, pred_b = probs_b.max(dim=-1, keepdim=True)
            pred_m = probs_m.argmax(dim=-1)

            soft_e = torch.matmul(probs_b, self.base_emb.weight) + torch.matmul(probs_m, self.mod_emb.weight)
            hard_e = self.base_emb(pred_b.squeeze(-1)) + self.mod_emb(pred_m)
            draft_e = hard_e - soft_e.detach() + soft_e

            # LLaDA / MaskGIT Progressive Top-k% + 75% Confidence Anchor Lock:
            # Locks >=75% confident tokens immediately AND guarantees progressive 35% -> 65%
            # anchor crystallization on open-ended AI `<think>` derivations!
            q_frac = 0.65 if k_pass == 1 else (0.35 if k_pass == 2 else 0.15)
            dyn_thr = torch.quantile(conf_b.detach(), q_frac, dim=1, keepdim=True).clamp(max=0.75)
            target_w = torch.where(conf_b >= dyn_thr, torch.ones_like(conf_b), conf_b)
            w = target_w - conf_b.detach() + conf_b
            y_locked = w * draft_e + (1.0 - w) * mask_emb

            z_k, bl, ml = self._denoise_pass(y_locked, z_history, mk, mv, p_state, pass_idx=k_pass)
            z_history.append(z_k)
            b_logs.append(bl)
            m_logs.append(ml)

        return b_logs, m_logs, b_mag, c_mag
