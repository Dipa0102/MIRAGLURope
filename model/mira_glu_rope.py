# -*- coding: utf-8 -*-
"""
MiraGLURopeLM — variante de CanalGlobalLM qui applique RoPE sur la partie FFN/GLU.

On reprend exactement l'équation de `model/canal_global.py` :

    z' = z  +  α_attn · A(z)  +  α_mlp · C( LN( −α_adv · ν · (Wu ⊙ Wg)(z) ) ) )

mais on modifie uniquement la branche GLU (Wu/Wg) en y appliquant une rotation
positionnelle RoPE *avant* le produit bilinéaire :

    z_glu = RoPE_base_glu(z) ; bilinear = Wu(z_glu) ⊙ Wg(z_glu)

**RoPE GLU** : une seule base **lente** (défaut 16180, `rope_base_glu_slow`) — rotation
quasi imperceptible d'une position à l'autre, compatible avec l'advection. Pas de
mélange fast/slow ni gate sur le GLU. L'**attention** conserve le RoPE hybride φ/1618
et le biais global (inchangés).
"""

from __future__ import annotations

# Base RoPE dédiée au GLU (indépendante de ROPE_BASE_SLOW=1618 de l'attention).
ROPE_BASE_GLU_DEFAULT = 16180.0

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["MiraGLURopeLM", "ROPE_BASE_GLU_DEFAULT"]

# ─────────────────────────────────────────────────────────────────────────────
# Dépendances intégrées (RoPE / biais global / viscosité)
# Objectif : dépôt lisible avec un seul fichier modèle.
# ─────────────────────────────────────────────────────────────────────────────

PHI = (1.0 + math.sqrt(5.0)) / 2.0
ROPE_BASE_FAST = float(PHI)
ROPE_BASE_SLOW = 1618.0


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _rope_cos_sin(
    seq_len: int,
    dim: int,
    base: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    d = dim - (dim % 2)
    if d == 0:
        z = torch.zeros(1, seq_len, 0, device=device, dtype=dtype)
        return z, z
    idx = torch.arange(0, d, 2, device=device, dtype=dtype) / float(d)
    inv = 1.0 / (base ** idx)
    t = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.outer(t, inv)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    return x * cos + _rotate_half(x) * sin


def apply_rope_hybrid(
    Q: torch.Tensor,
    K: torch.Tensor,
    gate: torch.Tensor,
    base_fast: float = ROPE_BASE_FAST,
    base_slow: float = ROPE_BASE_SLOW,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, L, D = Q.shape
    dev, dt = Q.device, Q.dtype
    cos_f, sin_f = _rope_cos_sin(L, D, base_fast, dev, dt)
    cos_s, sin_s = _rope_cos_sin(L, D, base_slow, dev, dt)
    cos = gate * cos_f + (1.0 - gate) * cos_s
    sin = gate * sin_f + (1.0 - gate) * sin_s
    return _apply_rope(Q, cos, sin), _apply_rope(K, cos, sin)


class DuplexViscosity(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.nu_diff = nn.Parameter(torch.tensor(0.2))
        self.nu_adv = nn.Parameter(torch.tensor(0.01))
        h = max(4, dim // 8)
        self.proj = nn.Sequential(nn.Linear(dim, h), nn.Tanh(), nn.Linear(h, 1))
        for m in self.proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, pool: torch.Tensor) -> torch.Tensor:
        delta = torch.tanh(self.proj(pool)) * self.nu_adv.abs()
        return self.nu_diff.abs() + delta


class GlobalCoherenceBias(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.scale = dim**-0.5
        self.W_coh = nn.Linear(dim, dim, bias=False)
        nn.init.normal_(self.W_coh.weight, std=0.02)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, global_vec: torch.Tensor, K_raw: torch.Tensor) -> torch.Tensor:
        gk = self.W_coh(global_vec)
        b = torch.matmul(K_raw, gk.unsqueeze(-1)).squeeze(-1) * self.scale
        return torch.tanh(self.gamma) * b.unsqueeze(1)


def _causal_cummean(z: torch.Tensor) -> torch.Tensor:
    """Moyenne cumulative causale : pool[t] = mean(z[0..t]). Shape (B,L,D)."""
    B, L, _ = z.shape
    c = z.cumsum(dim=1)
    denom = torch.arange(1, L + 1, device=z.device, dtype=z.dtype).view(1, L, 1)
    return c / denom


class KernelCouplingA_CausalPool(nn.Module):
    """
    Variante de KernelCouplingA qui reçoit un pool **par position** (B,L,D),
    pour éviter toute fuite de causalité due à un pool global calculé sur toute
    la fenêtre.

    Paramètres/poids volontairement alignés avec KernelCouplingA (W_q/W_k/W_v,
    rope_gate, global_bias) pour compat de state_dict.
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 1,
        rope_base_fast: float = ROPE_BASE_FAST,
        rope_base_slow: float = ROPE_BASE_SLOW,
    ):
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"KernelCouplingA : dim ({dim}) doit être divisible par n_heads ({n_heads}).")
        self.n_heads = int(n_heads)
        self.head_dim = dim // self.n_heads
        self.scale = self.head_dim ** -0.5
        self.rope_base_fast = float(rope_base_fast)
        self.rope_base_slow = float(rope_base_slow)

        self.W_q = nn.Linear(dim, dim, bias=True)
        self.W_k = nn.Linear(dim, dim, bias=True)
        self.W_v = nn.Linear(dim, dim, bias=True)
        for m in (self.W_q, self.W_k, self.W_v):
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

        h = max(8, dim // 4)
        self.rope_gate = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Linear(h, 1))
        for m in self.rope_gate.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.global_bias = GlobalCoherenceBias(dim)

    def forward(self, z: torch.Tensor, pool_seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        z        : (B,L,D)
        pool_seq : (B,L,D) — pool causal par position query
        """
        B, L, D = z.shape
        nh = self.n_heads
        hd = self.head_dim

        Q = self.W_q(z)  # (B,L,D)
        K_raw = self.W_k(z)  # (B,L,D)
        V = self.W_v(z)  # (B,L,D)

        # Biais global par query position : (B,L,L)
        gk = self.global_bias.W_coh(pool_seq)  # (B,L,D)
        gb = torch.matmul(gk, K_raw.transpose(1, 2)) * self.global_bias.scale  # (B,L,L)
        gb = torch.tanh(self.global_bias.gamma) * gb  # broadcast (1,)

        # RoPE hybride par query position (gate dépend de pool_seq)
        gate = torch.sigmoid(self.rope_gate(pool_seq)).view(B, L, 1)  # (B,L,1)
        Q, K_rot = apply_rope_hybrid(
            Q,
            K_raw,
            gate,
            base_fast=self.rope_base_fast,
            base_slow=self.rope_base_slow,
        )

        Q_h = Q.view(B, L, nh, hd).transpose(1, 2)  # (B,nh,L,hd)
        K_h = K_rot.view(B, L, nh, hd).transpose(1, 2)
        V_h = V.view(B, L, nh, hd).transpose(1, 2)

        logits = torch.matmul(Q_h, K_h.transpose(-2, -1)) * self.scale  # (B,nh,L,L)
        logits = logits + gb.unsqueeze(1)  # (B,1,L,L)

        mask = torch.triu(torch.ones(L, L, device=z.device, dtype=torch.bool), diagonal=1)
        logits = logits.masked_fill(mask.view(1, 1, L, L), float("-inf"))

        attn_w_h = torch.softmax(logits, dim=-1)
        out_h = torch.matmul(attn_w_h, V_h)
        out = out_h.transpose(1, 2).contiguous().view(B, L, D)
        attn_w = attn_w_h.mean(dim=1)
        return out, attn_w


class SequenceLayerGLURope(nn.Module):
    """Comme `SequenceLayer`, mais RoPE (base lente) sur la branche GLU uniquement."""

    def __init__(
        self,
        dim: int,
        n_heads: int = 1,
        alpha_attn: float = 0.42,
        alpha_mlp: float = 1.07,
        alpha_adv: float = 1.0,
        pre_ln: bool = False,
        residual_scale: float = 1.0,
        rope_base_fast: float = ROPE_BASE_FAST,
        rope_base_slow: float = ROPE_BASE_SLOW,
        rope_base_glu_fast: float = ROPE_BASE_FAST,
        rope_base_glu_slow: float = ROPE_BASE_GLU_DEFAULT,
    ):
        super().__init__()
        self.alpha_attn = float(alpha_attn)
        self.alpha_mlp = float(alpha_mlp)
        self.alpha_adv = float(alpha_adv)
        self.pre_ln = bool(pre_ln)
        self.residual_scale = float(residual_scale)
        # Une seule base pour le GLU : lente (défaut 16180), pas φ.
        self.rope_base_glu = float(rope_base_glu_slow)

        self.kernel_A = KernelCouplingA_CausalPool(
            dim,
            n_heads=n_heads,
            rope_base_fast=rope_base_fast,
            rope_base_slow=rope_base_slow,
        )
        self.nu = DuplexViscosity(dim)
        self.W_u = nn.Linear(dim, dim, bias=True)
        self.W_g = nn.Linear(dim, dim, bias=True)
        self.ln = nn.LayerNorm(dim)
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.C = nn.Linear(dim, dim, bias=True)

        for m in (self.W_u, self.W_g):
            nn.init.normal_(m.weight, std=0.02)
            nn.init.zeros_(m.bias)

        out_std = 0.02 * (self.residual_scale**0.5) if self.pre_ln else 0.02
        nn.init.normal_(self.C.weight, std=out_std)
        nn.init.zeros_(self.C.bias)

        if self.pre_ln:
            with torch.no_grad():
                self.kernel_A.W_v.weight.mul_(self.residual_scale**0.5)

    def _rope_glu(self, z: torch.Tensor) -> torch.Tensor:
        """RoPE à base fixe (lente) sur la branche GLU (pas de mélange ni gate)."""
        _B, L, D = z.shape
        cos, sin = _rope_cos_sin(L, D, self.rope_base_glu, z.device, z.dtype)
        return _apply_rope(z, cos, sin)

    def _forward_legacy(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pool_seq = _causal_cummean(z)  # (B,L,D) causal
        attn_out, attn_w = self.kernel_A(z, pool_seq)
        nu_b = self.nu(pool_seq.reshape(-1, z.size(-1))).view(z.size(0), z.size(1), 1)

        z_glu = self._rope_glu(z)
        bilinear = self.W_u(z_glu) * self.W_g(z_glu)
        adv = -(self.alpha_adv * nu_b) * bilinear

        mlp_out = self.C(self.ln(adv))
        z_next = z + self.alpha_attn * attn_out + self.alpha_mlp * mlp_out
        return z_next, attn_w

    def _forward_preln(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = self.residual_scale

        z1 = self.ln1(z)
        pool1_seq = _causal_cummean(z1)
        attn_out, attn_w = self.kernel_A(z1, pool1_seq)
        z = z + (scale * self.alpha_attn) * attn_out

        z2 = self.ln2(z)
        pool2_seq = _causal_cummean(z2)
        nu_b = self.nu(pool2_seq.reshape(-1, z2.size(-1))).view(z2.size(0), z2.size(1), 1)

        z2_glu = self._rope_glu(z2)
        bilinear = self.W_u(z2_glu) * self.W_g(z2_glu)
        adv = -(self.alpha_adv * nu_b) * bilinear

        mlp_out = self.C(adv)
        z = z + (scale * self.alpha_mlp) * mlp_out
        return z, attn_w

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pre_ln:
            return self._forward_preln(z)
        return self._forward_legacy(z)


class SequenceModelGLURope(nn.Module):
    """Empilement de N couches `SequenceLayerGLURope` (mêmes options que SequenceModel)."""

    def __init__(
        self,
        dim: int,
        n_layers: int = 2,
        n_heads: int = 1,
        pre_ln: bool = False,
        residual_scale: Optional[float] = None,
        rope_base_fast: float = ROPE_BASE_FAST,
        rope_base_slow: float = ROPE_BASE_SLOW,
        rope_base_glu_fast: float = ROPE_BASE_FAST,
        rope_base_glu_slow: float = ROPE_BASE_GLU_DEFAULT,
        **layer_kwargs,
    ):
        super().__init__()
        self.n_heads = int(n_heads)
        self.pre_ln = bool(pre_ln)
        if residual_scale is None:
            residual_scale = (1.0 / math.sqrt(2 * n_layers)) if pre_ln else 1.0
        self.residual_scale = float(residual_scale)
        self.layers = nn.ModuleList(
            [
                SequenceLayerGLURope(
                    dim,
                    n_heads=n_heads,
                    pre_ln=self.pre_ln,
                    residual_scale=self.residual_scale,
                    rope_base_fast=rope_base_fast,
                    rope_base_slow=rope_base_slow,
                    rope_base_glu_fast=rope_base_glu_fast,
                    rope_base_glu_slow=rope_base_glu_slow,
                    **layer_kwargs,
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            z, _ = layer(z)
        return z


class MiraGLURopeLM(nn.Module):
    """
    Modèle de langage BPE causal, équivalent à `CanalGlobalLM`, avec RoPE sur la branche GLU.

    Notes :
    - RoPE **GLU** : base **lente** (`rope_base_glu_slow`, défaut 16180). `rope_base_glu_fast`
      reste un paramètre distinct (souvent φ) pour métadonnées / CLI alignée sur l'attention.
    - Le vocab/tokenizer et les têtes sont identiques au CanalGlobalLM.
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        embed_dim: int = 256,
        n_layers: int = 6,
        n_heads: int = 8,
        alpha_attn: float = 0.42,
        alpha_mlp: float = 1.07,
        alpha_adv: float = 1.0,
        pre_ln: bool = False,
        rope_base_fast: float = ROPE_BASE_FAST,
        rope_base_slow: float = ROPE_BASE_SLOW,
        rope_base_glu_fast: float = ROPE_BASE_FAST,
        rope_base_glu_slow: float = ROPE_BASE_GLU_DEFAULT,
    ):
        super().__init__()
        self.seq_len = int(seq_len)
        self.embed_dim = int(embed_dim)
        self.n_heads = int(n_heads)
        self.pre_ln = bool(pre_ln)
        self.rope_base_fast = float(rope_base_fast)
        self.rope_base_slow = float(rope_base_slow)
        self.rope_base_glu = float(rope_base_glu_slow)
        self.rope_base_glu_fast = float(rope_base_glu_fast)
        self.rope_base_glu_slow = float(rope_base_glu_slow)

        self.tok_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(seq_len, embed_dim)
        self.body = SequenceModelGLURope(
            embed_dim,
            n_layers,
            n_heads=n_heads,
            alpha_attn=alpha_attn,
            alpha_mlp=alpha_mlp,
            alpha_adv=alpha_adv,
            pre_ln=self.pre_ln,
            rope_base_fast=rope_base_fast,
            rope_base_slow=rope_base_slow,
            rope_base_glu_fast=rope_base_glu_fast,
            rope_base_glu_slow=rope_base_glu_slow,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        B, L = token_ids.shape
        pos = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, -1)
        z = self.tok_emb(token_ids) + self.pos_emb(pos)
        z = self.body(z)
        z = self.norm(z)
        return self.head(z)

    @property
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    torch.manual_seed(0)
    m = MiraGLURopeLM(vocab_size=8000, seq_len=128, embed_dim=64, n_layers=2, n_heads=1).eval()
    x = torch.randint(0, 8000, (2, 64))
    with torch.no_grad():
        y = m(x)
    print("[OK] MiraGLURopeLM", tuple(y.shape), f"{m.n_params:,} params")
