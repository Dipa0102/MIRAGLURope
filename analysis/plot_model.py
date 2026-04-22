#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Analyse chirurgicale complète du Canal Global LM (MIRAZOOM).

Génère un ensemble de figures matplotlib détaillant l'état interne du modèle
couche par couche, matrice par matrice. Chaque figure est accompagnée d'un
rapport console qui explique ce qu'on regarde et ce que cela signifie.

Usage :
    python analysis/plot_model.py
    python analysis/plot_model.py --checkpoint checkpoint/mon_modele.pt

Les dimensions (embed_dim, n_layers, n_heads, vocab_size, seq_len) sont
détectées automatiquement depuis le checkpoint. Aucun hardcoding.

Figures produites dans analysis/figures/ :
    01_architecture.png         schéma conceptuel du modèle
    02_gamma_per_layer.png      γ et tanh(γ) de GlobalCoherenceBias par couche
    03_weight_spectra.png       spectres SVD de W_q / W_k / W_v / W_coh
    04_attention_entropy.png    entropie d'attention par couche × phrase
    05_latent_norms.png         évolution de ||z|| à travers les couches
    06_global_bias_per_token.png biais global par token et par couche
    07_embedding_space.png      PCA + spectre SVD des embeddings
    08_viscosity.png            ν_diff, ν_adv, ν effectif
    09_weight_distributions.png histogrammes de tous les poids
    10_head_weights.png         head Linear(D→V) : heatmap + biais + SVD
    11_multihead_attention.png  attention par tête (8 têtes, couche 0 et dernière)
    12_layer_contributions.png  ||z_{l+1} - z_l|| par couche (activité)
    13_rope_gate.png            activation du gate RoPE par couche × prompt

Sortie aussi : analysis/figures/stats.json (toutes les stats extraites).
"""

from __future__ import annotations

import argparse
import json
import math
import sys

# Force UTF-8 output on Windows consoles (cp1252 par défaut casse γ, →, etc.)
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch

# ══════════════════════════════════════════════════════════════════════════════
# Chemins et configuration globale
# ══════════════════════════════════════════════════════════════════════════════

ROOT     = Path(__file__).resolve().parent.parent
FIG_DIR  = Path(__file__).resolve().parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(ROOT))


# ── Palette graphique cohérente ────────────────────────────────────────────────

PALETTE = {
    "blue":   "#4C72B0",
    "orange": "#DD8452",
    "green":  "#55A868",
    "red":    "#C44E52",
    "purple": "#8172B2",
    "brown":  "#937860",
    "teal":   "#2A9D8F",
    "gold":   "#E9C46A",
}
LAYER_COLORS = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"],
                PALETTE["red"],  PALETTE["purple"], PALETTE["brown"],
                PALETTE["teal"], PALETTE["gold"]]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#FAFAFA",
    "axes.grid":        True,
    "grid.alpha":       0.35,
    "font.size":        10,
    "axes.titlesize":   12,
    "axes.labelsize":   10,
    "legend.fontsize":  9,
    "figure.titlesize": 14,
})


# ── Prompts de diagnostic (conversationnels, alignés sur le corpus propre) ────

DIAG_PROMPTS = [
    "<|user|> Bonjour, comment vas-tu ? <|endofturn|>\n<|mira|>",
    "<|user|> Peux-tu m'expliquer un concept scientifique ? <|endofturn|>\n<|mira|>",
    "<|user|> J'aimerais parler de cuisine aujourd'hui. <|endofturn|>\n<|mira|>",
    "<|user|> Comment gérer le stress au quotidien ? <|endofturn|>\n<|mira|>",
    "<|user|> Raconte-moi une histoire courte. <|endofturn|>\n<|mira|>",
    "<|user|> Quel est ton avis sur la musique classique ? <|endofturn|>\n<|mira|>",
]


# ══════════════════════════════════════════════════════════════════════════════
# Chargement du modèle
# ══════════════════════════════════════════════════════════════════════════════

def find_checkpoint(explicit: str = "") -> Path:
    """Retourne un chemin de checkpoint valide (argument explicite ou plus récent)."""
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            sys.exit(f"[ERREUR] Checkpoint introuvable : {p}")
        return p
    ckpt_dir = ROOT / "checkpoint"
    if not ckpt_dir.is_dir():
        sys.exit(f"[ERREUR] Dossier checkpoint/ introuvable : {ckpt_dir}")
    # On exclut les *_OLD_*.pt
    cands = sorted(
        (f for f in ckpt_dir.glob("*.pt")
         if "_OLD" not in f.name.upper() and "POISON" not in f.name.upper()),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    if not cands:
        sys.exit(f"[ERREUR] Aucun checkpoint .pt trouvé dans {ckpt_dir}")
    return cands[0]


def load_model_and_tokenizer(ckpt_path: Path) -> Tuple[Any, Any, Dict[str, Any]]:
    """Charge le modèle avec auto-détection des dimensions depuis le checkpoint."""
    from tokenizers import Tokenizer
    from model.mira_glu_rope import MiraGLURopeLM

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck.get("state_dict", ck)

    # Auto-détection depuis les tenseurs
    vocab_size = sd["tok_emb.weight"].shape[0]
    embed_dim  = sd["tok_emb.weight"].shape[1]
    seq_len    = sd["pos_emb.weight"].shape[0]

    # n_layers : compte les clés body.layers.N.*
    layer_ids = set()
    for k in sd.keys():
        if k.startswith("body.layers."):
            layer_ids.add(int(k.split(".")[2]))
    n_layers = max(layer_ids) + 1 if layer_ids else 0

    n_heads = int(ck.get("n_heads", 8))
    pre_ln = bool(ck.get("pre_ln", False))

    # Construction et chargement
    model = MiraGLURopeLM(
        vocab_size=vocab_size,
        seq_len=seq_len,
        embed_dim=embed_dim,
        n_layers=n_layers,
        n_heads=n_heads,
        pre_ln=pre_ln,
        rope_base_fast=float(ck.get("rope_base_fast", 1.6180)),
        rope_base_slow=float(ck.get("rope_base_slow", 1618.0)),
        rope_base_glu_fast=float(ck.get("rope_base_glu_fast", 1.6180)),
        rope_base_glu_slow=float(ck.get("rope_base_glu_slow", 16180.0)),
    )
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError as e:
        print(f"[AVERTISSEMENT] Chargement non strict : {e}")
        model.load_state_dict(sd, strict=False)
    model.eval()

    # Tokenizer : soit dans ck["tokenizer_path"], soit tokenizer/tokenizer.json
    tok_path = Path(str(ck.get("tokenizer_path") or "tokenizer/tokenizer_v3_fr12k.json"))
    if not tok_path.is_absolute():
        tok_path = ROOT / tok_path
    if not tok_path.is_file():
        sys.exit(f"[ERREUR] Tokenizer introuvable : {tok_path}")
    tokenizer = Tokenizer.from_file(str(tok_path))

    meta = {
        "ckpt_path":  str(ckpt_path),
        "ckpt_name":  ckpt_path.name,
        "vocab_size": vocab_size,
        "embed_dim":  embed_dim,
        "seq_len":    seq_len,
        "n_layers":   n_layers,
        "n_heads":    n_heads,
        "head_dim":   embed_dim // n_heads,
        "epochs_completed": ck.get("epochs_completed") or ck.get("epochs", "?"),
        "n_params":   sum(p.numel() for p in model.parameters()),
        "tokenizer_path": str(tok_path),
    }
    return model, tokenizer, meta


# ══════════════════════════════════════════════════════════════════════════════
# Extraction de statistiques
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ModelStats:
    meta:               Dict[str, Any]
    layer_gammas_raw:   List[float]       = field(default_factory=list)
    layer_gammas_tanh:  List[float]       = field(default_factory=list)
    layer_w_coh_norms:  List[float]       = field(default_factory=list)
    layer_nu_diff:      List[float]       = field(default_factory=list)
    layer_nu_adv:       List[float]       = field(default_factory=list)
    layer_alpha_attn:   List[float]       = field(default_factory=list)
    layer_alpha_mlp:    List[float]       = field(default_factory=list)
    layer_alpha_adv:    List[float]       = field(default_factory=list)
    matrix_norms:       Dict[str, float]  = field(default_factory=dict)
    matrix_ranks:       Dict[str, int]    = field(default_factory=dict)
    emb_spectral_entropy: float           = 0.0
    emb_effective_rank:  int              = 0
    head_effective_rank: int              = 0

    def to_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        return d


def extract_stats(model) -> ModelStats:
    """Parcourt tous les paramètres et extrait les statistiques clés."""
    stats = ModelStats(meta={})

    for i, layer in enumerate(model.body.layers):
        kA = layer.kernel_A
        gamma_raw = float(kA.global_bias.gamma.item())
        stats.layer_gammas_raw.append(gamma_raw)
        stats.layer_gammas_tanh.append(math.tanh(gamma_raw))
        stats.layer_w_coh_norms.append(float(kA.global_bias.W_coh.weight.norm().item()))

        stats.layer_nu_diff.append(float(layer.nu.nu_diff.abs().item()))
        stats.layer_nu_adv.append(float(layer.nu.nu_adv.abs().item()))
        stats.layer_alpha_attn.append(float(layer.alpha_attn))
        stats.layer_alpha_mlp.append(float(layer.alpha_mlp))
        stats.layer_alpha_adv.append(float(layer.alpha_adv))

        # Normes + rang effectif par matrice
        for name, W in [("W_q", kA.W_q.weight), ("W_k", kA.W_k.weight),
                        ("W_v", kA.W_v.weight),
                        ("W_coh", kA.global_bias.W_coh.weight),
                        ("W_u", layer.W_u.weight), ("W_g", layer.W_g.weight),
                        ("C",   layer.C.weight)]:
            W_d = W.detach()
            key = f"L{i}.{name}"
            stats.matrix_norms[key] = float(W_d.norm().item())
            try:
                svd = torch.linalg.svdvals(W_d).numpy()
                rank_eff = int((svd > 0.01 * svd[0]).sum())
                stats.matrix_ranks[key] = rank_eff
            except Exception:
                stats.matrix_ranks[key] = -1

    # Embeddings
    emb = model.tok_emb.weight.detach().numpy()
    _, S, _ = np.linalg.svd(emb - emb.mean(axis=0), full_matrices=False)
    s_norm = S / S.sum()
    stats.emb_spectral_entropy = float(-(s_norm * np.log(s_norm + 1e-12)).sum())
    stats.emb_effective_rank = int((S > 0.01 * S[0]).sum())

    # Head
    Wh = model.head.weight.detach().numpy()
    sv_h = np.linalg.svd(Wh, compute_uv=False)
    stats.head_effective_rank = int((sv_h > 0.01 * sv_h[0]).sum())

    return stats


# ══════════════════════════════════════════════════════════════════════════════
# Forward exploratoire (récupère z à chaque couche + attention)
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def forward_with_diagnostics(model, token_ids: torch.Tensor) -> Dict[str, Any]:
    """Forward avec collecte des z intermédiaires, attention weights, gates RoPE."""
    B, L = token_ids.shape
    pos = torch.arange(L, device=token_ids.device).unsqueeze(0).expand(B, -1)
    z = model.tok_emb(token_ids) + model.pos_emb(pos)

    diag = {
        "z_layers": [z.clone()],
        "attn_weights": [],
        "global_bias": [],
        "rope_gates":   [],
        "nu_effective": [],
    }

    for layer in model.body.layers:
        pool = z.mean(dim=1)

        # Biais global
        K_raw = layer.kernel_A.W_k(z)
        gb = layer.kernel_A.global_bias(pool, K_raw).squeeze(1)   # (B, L)
        diag["global_bias"].append(gb.clone())

        # RoPE gate
        gate = torch.sigmoid(layer.kernel_A.rope_gate(pool)).view(B, 1, 1)
        diag["rope_gates"].append(float(gate.mean().item()))

        # Viscosité
        diag["nu_effective"].append(float(layer.nu(pool).mean().item()))

        # Forward
        z_new, attn_w = layer(z)
        diag["attn_weights"].append(attn_w.clone())
        diag["z_layers"].append(z_new.clone())
        z = z_new

    return diag


# ══════════════════════════════════════════════════════════════════════════════
# FIG 01 — Architecture (schéma conceptuel, adapté aux dims détectées)
# ══════════════════════════════════════════════════════════════════════════════

def plot_01_architecture(meta: Dict[str, Any]):
    fig, ax = plt.subplots(figsize=(11, 9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    def box(x, y, w, h, color, label, fs=10, bold=False, alpha=0.9):
        r = mpatches.FancyBboxPatch((x-w/2, y-h/2), w, h,
            boxstyle="round,pad=0.15", facecolor=color, edgecolor="#333",
            linewidth=1.2, alpha=alpha, zorder=3)
        ax.add_patch(r)
        ax.text(x, y, label, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", zorder=4)

    def arrow(x1, y1, x2, y2, color="#555"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5), zorder=5)

    # Entrée
    box(5, 9.4, 4.0, 0.55, "#AED6F1", f"token_ids  (B, {meta['seq_len']})", fs=10)
    arrow(5, 9.1, 5, 8.75)
    box(5, 8.45, 4.5, 0.55, "#AED6F1",
        f"tok_emb + pos_emb  →  z  (B, {meta['seq_len']}, {meta['embed_dim']})", fs=9)
    arrow(5, 8.15, 5, 7.75)

    # Couches empilées
    n = meta["n_layers"]
    y_top = 7.4
    y_bot = 2.3
    if n > 1:
        step = (y_top - y_bot) / max(1, n - 1) if n > 1 else 0
    else:
        step = 0
    heights = [y_top - i * step if n > 1 else (y_top + y_bot) / 2 for i in range(n)]

    for i, y in enumerate(heights):
        col = LAYER_COLORS[i % len(LAYER_COLORS)]
        box(5, y, 5.5, 0.55,
            "#E8DAEF" if i % 2 == 0 else "#FDEBD0",
            f"SequenceLayer [couche {i}]   attention {meta['n_heads']} têtes  |  head_dim={meta['head_dim']}",
            fs=9, bold=True)
        if i < n - 1:
            arrow(5, y - 0.3, 5, heights[i+1] + 0.3)

    # Sortie
    arrow(5, heights[-1] - 0.3, 5, 1.75)
    box(5, 1.45, 4.0, 0.55, "#AED6F1", "LayerNorm finale", fs=9)
    arrow(5, 1.15, 5, 0.75)
    box(5, 0.45, 4.5, 0.55, "#AED6F1",
        f"head  Linear({meta['embed_dim']} → {meta['vocab_size']})  →  logits", fs=9)

    # Légende gauche : composants internes d'une couche
    ax.text(1.3, 7.4, "Dans chaque couche :", ha="center", fontsize=10, fontweight="bold",
            color="#1a5276")
    inner = [
        "KernelCouplingA",
        "  • Q, K, V  (Linear D→D)",
        "  • RoPE hybride (φ, 1618)",
        "  • GlobalCoherenceBias :",
        "    bias = tanh(γ) · ⟨W_coh(pool), K⟩/√d",
        "  • Attention multi-tête softmax",
        "",
        "Branche bilinéaire :",
        "  • ν = DuplexViscosity(pool)",
        "  • adv = −α_adv · ν · (Wu ⊙ Wg)(z)",
        "  • C(LayerNorm(adv))",
        "",
        "Résidu :",
        "  z' = z + α_attn · A + α_mlp · C(LN(adv))",
    ]
    for j, line in enumerate(inner):
        ax.text(1.3, 7.0 - j*0.32, line, ha="center", va="center", fontsize=8.2,
                color="#1a1a1a")

    # Légende droite : hyperparams
    box(8.6, 6.0, 2.5, 3.5, "#FDFEFE", "", fs=8, alpha=0.5)
    ax.text(8.6, 7.5, "Hyperparamètres", ha="center", fontsize=10, fontweight="bold",
            color="#1a5276")
    hp = [
        f"vocab     = {meta['vocab_size']:>6d}",
        f"seq_len   = {meta['seq_len']:>6d}",
        f"embed_dim = {meta['embed_dim']:>6d}",
        f"n_layers  = {meta['n_layers']:>6d}",
        f"n_heads   = {meta['n_heads']:>6d}",
        f"head_dim  = {meta['head_dim']:>6d}",
        "",
        f"Params    = {meta['n_params']:>9,d}",
        f"Epochs    = {meta['epochs_completed']}",
    ]
    for j, line in enumerate(hp):
        ax.text(8.6, 7.2 - j*0.32, line, ha="center", va="center", fontsize=8.3,
                family="monospace")

    ax.set_title(f"Architecture — Canal Global LM  ({meta['ckpt_name']})",
                 fontsize=13, fontweight="bold", pad=8)
    fig.tight_layout()
    out = FIG_DIR / "01_architecture.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# FIG 02 — γ et tanh(γ) par couche
# ══════════════════════════════════════════════════════════════════════════════

def plot_02_gamma(stats: ModelStats):
    n = len(stats.layer_gammas_raw)
    x = np.arange(n)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # γ brut
    colors = ["#C44E52" if g > 0 else "#4C72B0" for g in stats.layer_gammas_raw]
    axes[0].bar(x, stats.layer_gammas_raw, color=colors, alpha=0.85, edgecolor="#333")
    axes[0].axhline(0, color="#999", lw=1, ls="--")
    axes[0].axhline( 0.245, color=PALETTE["green"], lw=1, ls=":", label="clamp ±0.245")
    axes[0].axhline(-0.245, color=PALETTE["green"], lw=1, ls=":")
    axes[0].set_xticks(x); axes[0].set_xticklabels([f"c{i}" for i in range(n)])
    axes[0].set_ylabel(r"$\gamma$ brut (paramètre appris)")
    axes[0].set_title(r"$\gamma$ par couche  [clamp $|\gamma| \leq 0.245$ pendant training]")
    axes[0].legend()

    # tanh(γ) = intensité effective du biais
    intens = [t * 100 for t in stats.layer_gammas_tanh]
    colors2 = ["#C44E52" if v > 0 else "#4C72B0" for v in intens]
    axes[1].bar(x, intens, color=colors2, alpha=0.85, edgecolor="#333")
    axes[1].axhline(0, color="#999", lw=1, ls="--")
    axes[1].axhline( 24, color=PALETTE["green"], lw=1, ls=":", label="borne ±24 %")
    axes[1].axhline(-24, color=PALETTE["green"], lw=1, ls=":")
    axes[1].set_xticks(x); axes[1].set_xticklabels([f"c{i}" for i in range(n)])
    axes[1].set_ylabel(r"$\tanh(\gamma) \times 100$  [%]")
    axes[1].set_title("Intensité effective du biais global par couche")
    axes[1].legend()

    fig.suptitle("GlobalCoherenceBias — paramètre de cohérence globale appris",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = FIG_DIR / "02_gamma_per_layer.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# FIG 03 — Spectres SVD W_q / W_k / W_v / W_coh pour toutes les couches
# ══════════════════════════════════════════════════════════════════════════════

def plot_03_weight_spectra(model, stats: ModelStats):
    n = len(model.body.layers)
    matrices_names = ["W_q", "W_k", "W_v", "W_coh"]
    fig, axes = plt.subplots(n, 4, figsize=(14, 2.4 * n + 1))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i, layer in enumerate(model.body.layers):
        kA = layer.kernel_A
        mats = [
            ("W_q",   kA.W_q.weight.detach()),
            ("W_k",   kA.W_k.weight.detach()),
            ("W_v",   kA.W_v.weight.detach()),
            ("W_coh", kA.global_bias.W_coh.weight.detach()),
        ]
        for j, (name, W) in enumerate(mats):
            svd = torch.linalg.svdvals(W).numpy()
            ax = axes[i, j]
            ax.bar(range(len(svd)), svd, color=LAYER_COLORS[j % 8], alpha=0.75, width=1.0)
            if i == 0:
                ax.set_title(name, fontsize=11, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"couche {i}", fontsize=10, fontweight="bold")
            rank_eff = int((svd > 0.01 * svd[0]).sum())
            ax.text(0.97, 0.95, f"rang={rank_eff}/{len(svd)}\n‖W‖={float(W.norm()):.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
                family="monospace",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
            ax.tick_params(labelsize=7)

    fig.suptitle("Spectres SVD des matrices clés  (rang effectif : σᵢ > 1% σ_max)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = FIG_DIR / "03_weight_spectra.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# FIG 04 — Entropie d'attention par couche × phrase
# ══════════════════════════════════════════════════════════════════════════════

def plot_04_attention_entropy(model, tokenizer, stats: ModelStats):
    n = len(model.body.layers)
    rows: List[Dict[str, Any]] = []

    for phrase in DIAG_PROMPTS:
        ids = tokenizer.encode(phrase).ids[:model.seq_len]
        if not ids: continue
        x = torch.tensor([ids], dtype=torch.long)
        diag = forward_with_diagnostics(model, x)
        L = len(ids)
        logL = math.log(L) if L > 1 else 1.0
        entropies = []
        for aw in diag["attn_weights"]:
            H = float((-aw * (aw + 1e-9).log()).sum(dim=-1).mean().item())
            entropies.append(H)
        label = (phrase[5:25] + "…") if len(phrase) > 30 else phrase[5:]
        rows.append({"label": label, "L": L, "H": entropies, "logL": logL})

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # H absolue (grouped bar par couche)
    labels = [r["label"] for r in rows]
    x_pos = np.arange(len(labels))
    w = 0.85 / n
    for i in range(n):
        vals = [r["H"][i] for r in rows]
        axes[0].bar(x_pos + (i - n/2 + 0.5) * w, vals, w,
                    label=f"c{i}", color=LAYER_COLORS[i % 8], alpha=0.85)
    axes[0].set_xticks(x_pos); axes[0].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    axes[0].set_ylabel("Entropie H  (nats)")
    axes[0].set_title("Entropie d'attention absolue")
    axes[0].legend(ncol=min(n, 4), fontsize=8)

    # H / log(L) = entropie relative
    for i in range(n):
        vals = [r["H"][i] / r["logL"] * 100 for r in rows]
        axes[1].bar(x_pos + (i - n/2 + 0.5) * w, vals, w,
                    label=f"c{i}", color=LAYER_COLORS[i % 8], alpha=0.85)
    axes[1].axhline(100, color="#aaa", lw=1, ls=":", label="max (uniforme)")
    axes[1].set_xticks(x_pos); axes[1].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
    axes[1].set_ylabel("H / log(L)  [%]")
    axes[1].set_title("Entropie d'attention relative  (0=point, 100=uniforme)")
    axes[1].legend(ncol=min(n, 4), fontsize=8)

    fig.suptitle("Concentration de l'attention par couche et par phrase",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = FIG_DIR / "04_attention_entropy.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# FIG 05 — Évolution de ||z|| à travers les couches
# ══════════════════════════════════════════════════════════════════════════════

def plot_05_latent_norms(model, tokenizer):
    n = len(model.body.layers)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    all_norms = []
    for i, phrase in enumerate(DIAG_PROMPTS[:5]):
        ids = tokenizer.encode(phrase).ids[:model.seq_len]
        if not ids: continue
        x = torch.tensor([ids], dtype=torch.long)
        diag = forward_with_diagnostics(model, x)
        norms = [float(z.norm(dim=-1).mean().item()) for z in diag["z_layers"]]
        all_norms.append(norms)
        label = (phrase[5:28] + "…") if len(phrase) > 33 else phrase[5:]
        axes[0].plot(range(len(norms)), norms, "o-",
                     color=LAYER_COLORS[i % 8], lw=2, ms=6, label=label)

    axes[0].set_xticks(range(n + 1))
    axes[0].set_xticklabels(["emb"] + [f"c{i}" for i in range(n)], fontsize=9)
    axes[0].set_ylabel("‖z‖ moyen sur la séquence")
    axes[0].set_title("Norme latente ‖z‖ à travers les couches")
    axes[0].legend(fontsize=7)

    # Facteurs de croissance
    if all_norms:
        growth = np.array([[n_[i+1]/n_[i] for i in range(len(n_)-1)] for n_ in all_norms])
        mean_growth = growth.mean(axis=0)
        std_growth  = growth.std(axis=0)
        x_pos = np.arange(n)
        axes[1].bar(x_pos, mean_growth, yerr=std_growth, capsize=5,
                    color=[LAYER_COLORS[i % 8] for i in range(n)], alpha=0.85,
                    edgecolor="#333")
        axes[1].axhline(1.0, color="#aaa", lw=1, ls="--", label="stabilité")
        axes[1].set_xticks(x_pos); axes[1].set_xticklabels([f"c{i}" for i in range(n)])
        axes[1].set_ylabel(r"Facteur moyen $\|z_{n+1}\|/\|z_n\|$")
        axes[1].set_title("Amplification par couche (moyenne ± écart-type sur prompts)")
        axes[1].legend()

    fig.suptitle("Dynamique des normes latentes", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = FIG_DIR / "05_latent_norms.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# FIG 06 — Biais global par token et par couche (sur 1 prompt choisi)
# ══════════════════════════════════════════════════════════════════════════════

def plot_06_global_bias_per_token(model, tokenizer):
    phrase = DIAG_PROMPTS[0]
    ids = tokenizer.encode(phrase).ids[:model.seq_len]
    x = torch.tensor([ids], dtype=torch.long)
    diag = forward_with_diagnostics(model, x)
    toks = [tokenizer.decode([i]) for i in ids]

    n = len(model.body.layers)
    rows = (n + 1) // 2
    cols = 2
    fig, axes = plt.subplots(rows, cols, figsize=(15, 2.4 * rows + 1))
    axes = np.atleast_2d(axes)

    for i, gb in enumerate(diag["global_bias"]):
        r, c = divmod(i, cols)
        ax = axes[r, c]
        gb_np = gb.squeeze().numpy()   # (L,)
        colors = ["#C44E52" if v >= 0 else "#4C72B0" for v in gb_np]
        ax.bar(range(len(gb_np)), gb_np, color=colors, alpha=0.82, edgecolor="#333", lw=0.3)
        ax.axhline(0, color="#aaa", lw=0.8, ls="--")
        ax.set_xticks(range(len(toks)))
        ax.set_xticklabels(toks, rotation=50, ha="right", fontsize=7)
        g = float(model.body.layers[i].kernel_A.global_bias.gamma.item())
        ax.set_title(f"couche {i}   γ={g:+.3f}   tanh(γ)={math.tanh(g):+.3f}", fontsize=10)
        ax.set_ylabel("biais global  (logit)")

    # Cache le dernier axe vide si n impair
    if n < rows * cols:
        axes[rows-1, cols-1].axis("off")

    fig.suptitle(
        f"Biais global par token  —  prompt : « {phrase[:50]}… »\n"
        "rouge = attire (+)  |  bleu = repousse (−)",
        fontsize=12, fontweight="bold")
    fig.tight_layout()
    out = FIG_DIR / "06_global_bias_per_token.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# FIG 07 — Espace des embeddings (PCA 2D + spectre SVD)
# ══════════════════════════════════════════════════════════════════════════════

def plot_07_embedding_space(model, tokenizer, stats: ModelStats):
    emb = model.tok_emb.weight.detach().numpy()
    V, D = emb.shape

    # PCA via SVD centré
    U, S, _ = np.linalg.svd(emb - emb.mean(axis=0), full_matrices=False)
    pc1 = U[:, 0] * S[0]
    pc2 = U[:, 1] * S[1]
    pct_var = S**2 / (S**2).sum() * 100

    # Top tokens par norme d'embedding
    norms = np.linalg.norm(emb, axis=1)
    top_idx = norms.argsort()[-25:][::-1]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    sc = axes[0].scatter(pc1, pc2, c=norms, cmap="viridis", s=3, alpha=0.5)
    plt.colorbar(sc, ax=axes[0], label="‖embedding‖")
    axes[0].scatter(pc1[top_idx], pc2[top_idx], color=PALETTE["red"], s=30, zorder=5,
                    edgecolors="#333", linewidths=0.5)
    for idx in top_idx[:15]:
        tok = tokenizer.decode([int(idx)])
        axes[0].annotate(repr(tok)[:16], (pc1[idx], pc2[idx]), fontsize=7,
                         color="#8B0000", xytext=(4, 3), textcoords="offset points")
    axes[0].set_xlabel(f"PC1 ({pct_var[0]:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({pct_var[1]:.1f}%)")
    axes[0].set_title(f"PCA 2D des {V} embeddings  (PC1+PC2 = {pct_var[0]+pct_var[1]:.1f}%)")

    # Spectre
    show = min(D, 128)
    axes[1].bar(range(show), S[:show], color=PALETTE["blue"], alpha=0.75)
    axes[1].fill_between(range(show), S[:show], alpha=0.3, color=PALETTE["blue"])
    axes[1].set_xlabel("Composante principale")
    axes[1].set_ylabel("σ")
    axes[1].set_title(f"Spectre SVD — rang effectif {stats.emb_effective_rank}/{D}")
    axes[1].text(0.97, 0.95,
        f"entropie spectrale\n{stats.emb_spectral_entropy:.3f}/{math.log(D):.3f}\n\n"
        f"σ_max = {S[0]:.2f}\nσ_min = {S[-1]:.4f}",
        transform=axes[1].transAxes, ha="right", va="top", fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    fig.suptitle(f"Espace des embeddings BPE ({V} tokens, dim {D})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = FIG_DIR / "07_embedding_space.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# FIG 08 — Viscosité DuplexViscosity
# ══════════════════════════════════════════════════════════════════════════════

def plot_08_viscosity(model, tokenizer, stats: ModelStats):
    n = len(model.body.layers)
    nu_eff_per_prompt: List[List[float]] = []
    labels = []
    for phrase in DIAG_PROMPTS[:5]:
        ids = tokenizer.encode(phrase).ids[:model.seq_len]
        if not ids: continue
        x = torch.tensor([ids], dtype=torch.long)
        diag = forward_with_diagnostics(model, x)
        nu_eff_per_prompt.append(diag["nu_effective"])
        labels.append((phrase[5:25] + "…") if len(phrase) > 30 else phrase[5:])

    nu_arr = np.array(nu_eff_per_prompt)  # (n_prompts, n_layers)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ν effectif par couche × prompt (heatmap)
    im = axes[0].imshow(nu_arr, aspect="auto", cmap="plasma")
    plt.colorbar(im, ax=axes[0], label="ν effectif")
    axes[0].set_xticks(range(n)); axes[0].set_xticklabels([f"c{i}" for i in range(n)])
    axes[0].set_yticks(range(len(labels))); axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_title("ν effectif par couche et prompt")

    # ν_diff appris
    axes[1].bar(range(n), stats.layer_nu_diff,
                color=[LAYER_COLORS[i%8] for i in range(n)], alpha=0.85, edgecolor="#333")
    axes[1].set_xticks(range(n)); axes[1].set_xticklabels([f"c{i}" for i in range(n)])
    axes[1].set_ylabel(r"$|\nu_\mathrm{diff}|$")
    axes[1].set_title("ν_diff appris (composante diffusion)")

    # ν_adv appris
    axes[2].bar(range(n), stats.layer_nu_adv,
                color=[LAYER_COLORS[i%8] for i in range(n)], alpha=0.85, edgecolor="#333")
    axes[2].set_xticks(range(n)); axes[2].set_xticklabels([f"c{i}" for i in range(n)])
    axes[2].set_ylabel(r"$|\nu_\mathrm{adv}|$")
    axes[2].set_title("ν_adv appris (amplitude advection)")

    fig.suptitle("DuplexViscosity — contrôle de la branche bilinéaire",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = FIG_DIR / "08_viscosity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# FIG 09 — Histogrammes de tous les poids par sous-module
# ══════════════════════════════════════════════════════════════════════════════

def plot_09_weight_distributions(model):
    groups: Dict[str, List[torch.Tensor]] = {
        "tok_emb":  [model.tok_emb.weight.detach().flatten()],
        "pos_emb":  [model.pos_emb.weight.detach().flatten()],
        "W_q":      [], "W_k": [], "W_v": [], "W_coh": [],
        "W_u":      [], "W_g": [], "C":    [],
        "head":     [model.head.weight.detach().flatten()],
    }
    for layer in model.body.layers:
        groups["W_q"  ].append(layer.kernel_A.W_q.weight.detach().flatten())
        groups["W_k"  ].append(layer.kernel_A.W_k.weight.detach().flatten())
        groups["W_v"  ].append(layer.kernel_A.W_v.weight.detach().flatten())
        groups["W_coh"].append(layer.kernel_A.global_bias.W_coh.weight.detach().flatten())
        groups["W_u"  ].append(layer.W_u.weight.detach().flatten())
        groups["W_g"  ].append(layer.W_g.weight.detach().flatten())
        groups["C"    ].append(layer.C.weight.detach().flatten())

    n = len(groups)
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 2.8 * rows))
    axes = axes.flatten()

    for i, (name, tensors) in enumerate(groups.items()):
        if not tensors:
            axes[i].axis("off"); continue
        W = torch.cat(tensors).numpy()
        axes[i].hist(W, bins=100, color=LAYER_COLORS[i % 8], alpha=0.85, edgecolor="white", lw=0.2)
        axes[i].axvline(0, color="#aaa", lw=0.8, ls="--")
        axes[i].set_title(f"{name}  ({len(W):,} params)", fontsize=10, fontweight="bold")
        axes[i].text(0.97, 0.95,
            f"μ={W.mean():+.4f}\nσ={W.std():.4f}\nmin={W.min():+.2f}\nmax={W.max():+.2f}",
            transform=axes[i].transAxes, ha="right", va="top", fontsize=7.5,
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
        axes[i].tick_params(labelsize=7)

    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Distribution des poids par type de sous-module",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = FIG_DIR / "09_weight_distributions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# FIG 10 — Tête de prédiction
# ══════════════════════════════════════════════════════════════════════════════

def plot_10_head_weights(model, stats: ModelStats):
    W = model.head.weight.detach().numpy()   # (V, D)
    b = model.head.bias.detach().numpy()
    V, D = W.shape

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Heatmap des N premiers tokens
    N = min(80, V)
    im = axes[0].imshow(W[:N, :], aspect="auto", cmap="RdBu_r",
                        vmin=-W[:N].std()*2.5, vmax=W[:N].std()*2.5)
    plt.colorbar(im, ax=axes[0])
    axes[0].set_xlabel(f"Dimension cachée (0..{D-1})")
    axes[0].set_ylabel(f"Token ID (0..{N-1})")
    axes[0].set_title(f"W_head[:{N}, :]   (rouge=+, bleu=−)")

    # Biais
    axes[1].hist(b, bins=80, color=PALETTE["blue"], alpha=0.85, edgecolor="white")
    axes[1].axvline(b.mean(), color=PALETTE["red"], lw=2, ls="--", label=f"μ={b.mean():+.4f}")
    axes[1].axvline(0, color="#aaa", lw=1)
    axes[1].set_xlabel("Valeur du biais")
    axes[1].set_ylabel("Nombre de tokens")
    axes[1].set_title(f"Biais head  (σ={b.std():.4f},  plage=[{b.min():+.3f}, {b.max():+.3f}])")
    axes[1].legend()

    # SVD W_head
    sv = np.linalg.svd(W, compute_uv=False)
    axes[2].bar(range(len(sv)), sv, color=PALETTE["green"], alpha=0.8, width=1.0)
    axes[2].set_xlabel("Rang singulier")
    axes[2].set_ylabel("σ")
    axes[2].set_title(f"Spectre SVD W_head ({V}×{D})\nrang effectif = {stats.head_effective_rank}/{D}")
    axes[2].text(0.97, 0.95,
        f"σ_max = {sv[0]:.2f}\nσ_min = {sv[-1]:.4f}\nratio = {sv[0]/sv[-1]:.1f}",
        transform=axes[2].transAxes, ha="right", va="top", fontsize=9,
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))

    fig.suptitle(f"Tête de prédiction  head: Linear({D} → {V})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = FIG_DIR / "10_head_weights.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# FIG 11 — Multi-head attention (chaque tête visualisée) pour 1 prompt
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def plot_11_multihead_attention(model, tokenizer):
    phrase = DIAG_PROMPTS[0]
    ids = tokenizer.encode(phrase).ids[:model.seq_len]
    x = torch.tensor([ids], dtype=torch.long)
    toks = [tokenizer.decode([i]) for i in ids]
    L = len(ids)

    # Re-exécuter manuellement pour récupérer attn_weights_h par tête
    B = 1
    pos = torch.arange(L).unsqueeze(0)
    z = model.tok_emb(x) + model.pos_emb(pos)

    # On visualise 2 couches : première et dernière
    layer_idx_to_show = [0, len(model.body.layers) - 1]
    if layer_idx_to_show[0] == layer_idx_to_show[1]:
        layer_idx_to_show = [0]

    # Forward jusqu'à chaque couche choisie, collecte des attention par tête
    collected = {}
    for tgt in layer_idx_to_show:
        z_tmp = z.clone()
        for i, layer in enumerate(model.body.layers):
            if i < tgt:
                z_tmp, _ = layer(z_tmp)
                continue
            # Couche cible : reproduire l'attention par tête
            kA = layer.kernel_A
            pool = z_tmp.mean(dim=1)
            Q     = kA.W_q(z_tmp)
            K_raw = kA.W_k(z_tmp)
            V     = kA.W_v(z_tmp)
            gb    = kA.global_bias(pool, K_raw).unsqueeze(1)   # (B,1,1,L)
            gate  = torch.sigmoid(kA.rope_gate(pool)).view(B, 1, 1)
            from model.mira_glu_rope import apply_rope_hybrid
            Q, K_rot = apply_rope_hybrid(Q, K_raw, gate)
            nh, hd = kA.n_heads, kA.head_dim
            Qh = Q.view(B, L, nh, hd).transpose(1, 2)
            Kh = K_rot.view(B, L, nh, hd).transpose(1, 2)
            logits = torch.matmul(Qh, Kh.transpose(-2, -1)) * kA.scale + gb
            mask = torch.triu(torch.ones(L, L, dtype=torch.bool), diagonal=1)
            logits = logits.masked_fill(mask.view(1, 1, L, L), float("-inf"))
            aw = torch.softmax(logits, dim=-1)   # (B, nh, L, L)
            collected[tgt] = aw[0].numpy()  # (nh, L, L)
            break

    for layer_idx, aw_per_head in collected.items():
        nh = aw_per_head.shape[0]
        cols = min(4, nh)
        rows = (nh + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.0 * rows + 0.5))
        axes = np.atleast_2d(axes)
        for h in range(nh):
            r, c = divmod(h, cols)
            ax = axes[r, c]
            im = ax.imshow(aw_per_head[h], cmap="magma", aspect="auto", vmin=0, vmax=aw_per_head[h].max())
            ax.set_title(f"tête {h}", fontsize=9)
            ax.set_xticks(range(L)); ax.set_yticks(range(L))
            ax.set_xticklabels(toks, rotation=55, ha="right", fontsize=5.5)
            ax.set_yticklabels(toks, fontsize=5.5)
            ax.tick_params(labelsize=5)
        for j in range(nh, rows * cols):
            axes.flatten()[j].axis("off")

        fig.suptitle(
            f"Attention par tête — couche {layer_idx}  (n_heads={nh})\n"
            f"prompt : « {phrase} »",
            fontsize=12, fontweight="bold")
        fig.tight_layout()
        out = FIG_DIR / f"11_multihead_attention_c{layer_idx}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return FIG_DIR / "11_multihead_attention_c0.png"


# ══════════════════════════════════════════════════════════════════════════════
# FIG 12 — Contributions par couche (norme du delta)
# ══════════════════════════════════════════════════════════════════════════════

def plot_12_layer_contributions(model, tokenizer):
    n = len(model.body.layers)
    contribs_per_prompt: List[List[float]] = []
    labels = []
    for phrase in DIAG_PROMPTS[:5]:
        ids = tokenizer.encode(phrase).ids[:model.seq_len]
        if not ids: continue
        x = torch.tensor([ids], dtype=torch.long)
        diag = forward_with_diagnostics(model, x)
        z_list = diag["z_layers"]
        deltas = [float((z_list[i+1] - z_list[i]).norm(dim=-1).mean().item()) for i in range(n)]
        contribs_per_prompt.append(deltas)
        labels.append((phrase[5:25] + "…") if len(phrase) > 30 else phrase[5:])

    contribs = np.array(contribs_per_prompt)
    mean_c = contribs.mean(axis=0)
    std_c  = contribs.std(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Moyenne ± std
    axes[0].bar(range(n), mean_c, yerr=std_c, capsize=5,
                color=[LAYER_COLORS[i%8] for i in range(n)], alpha=0.85, edgecolor="#333")
    axes[0].set_xticks(range(n)); axes[0].set_xticklabels([f"c{i}" for i in range(n)])
    axes[0].set_ylabel(r"$\|z_{l+1} - z_l\|$ moyen")
    axes[0].set_title("Contribution moyenne par couche\n(moyenne ± std sur 5 prompts)")

    # Par prompt
    for i, (deltas, label) in enumerate(zip(contribs_per_prompt, labels)):
        axes[1].plot(range(n), deltas, "o-",
                     color=LAYER_COLORS[i % 8], lw=2, ms=6, label=label)
    axes[1].set_xticks(range(n)); axes[1].set_xticklabels([f"c{i}" for i in range(n)])
    axes[1].set_ylabel(r"$\|z_{l+1} - z_l\|$")
    axes[1].set_title("Contribution par couche et par prompt")
    axes[1].legend(fontsize=7)

    fig.suptitle("Activité de chaque couche  (plus haut = plus de travail)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = FIG_DIR / "12_layer_contributions.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# FIG 13 — Gate RoPE par couche × prompt
# ══════════════════════════════════════════════════════════════════════════════

def plot_13_rope_gate(model, tokenizer):
    n = len(model.body.layers)
    gates_per_prompt: List[List[float]] = []
    labels = []
    for phrase in DIAG_PROMPTS:
        ids = tokenizer.encode(phrase).ids[:model.seq_len]
        if not ids: continue
        x = torch.tensor([ids], dtype=torch.long)
        diag = forward_with_diagnostics(model, x)
        gates_per_prompt.append(diag["rope_gates"])
        labels.append((phrase[5:25] + "…") if len(phrase) > 30 else phrase[5:])

    gates = np.array(gates_per_prompt)  # (n_prompts, n_layers)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Heatmap
    im = axes[0].imshow(gates, aspect="auto", cmap="coolwarm", vmin=0, vmax=1)
    plt.colorbar(im, ax=axes[0], label="gate  (0=lente 1618  |  1=rapide φ)")
    axes[0].set_xticks(range(n)); axes[0].set_xticklabels([f"c{i}" for i in range(n)])
    axes[0].set_yticks(range(len(labels))); axes[0].set_yticklabels(labels, fontsize=8)
    axes[0].set_title("Gate RoPE appris par couche et prompt")

    # Moyenne par couche
    mean_g = gates.mean(axis=0)
    std_g  = gates.std(axis=0)
    axes[1].bar(range(n), mean_g, yerr=std_g, capsize=5,
                color=[LAYER_COLORS[i%8] for i in range(n)], alpha=0.85, edgecolor="#333")
    axes[1].axhline(0.5, color="#aaa", lw=1, ls="--", label="mélange 50/50")
    axes[1].set_xticks(range(n)); axes[1].set_xticklabels([f"c{i}" for i in range(n)])
    axes[1].set_ylabel("Gate moyen")
    axes[1].set_title("Préférence RoPE par couche\n(vers φ rapide ou 1618 lente)")
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    fig.suptitle("RoPE hybride  —  gate σ(MLP(mean(z))) appris",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = FIG_DIR / "13_rope_gate.png"
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return out


# ══════════════════════════════════════════════════════════════════════════════
# Rapport console
# ══════════════════════════════════════════════════════════════════════════════

def print_text_report(meta: Dict[str, Any], stats: ModelStats):
    print()
    print("=" * 78)
    print("  RAPPORT D'ANALYSE  —  Canal Global LM (MIRAZOOM)")
    print("=" * 78)
    print(f"  Checkpoint : {meta['ckpt_name']}")
    print(f"  Epochs     : {meta['epochs_completed']}")
    print(f"  Paramètres : {meta['n_params']:,}")
    print(f"  Architecture : V={meta['vocab_size']}  L={meta['seq_len']}  "
          f"D={meta['embed_dim']}  n_layers={meta['n_layers']}  "
          f"n_heads={meta['n_heads']}  head_dim={meta['head_dim']}")
    print("-" * 78)
    print("  Gamma par couche (GlobalCoherenceBias) :")
    for i, (g, gt) in enumerate(zip(stats.layer_gammas_raw, stats.layer_gammas_tanh)):
        w_coh = stats.layer_w_coh_norms[i]
        role = "-> attire"  if gt >  0.02 else ("-> repousse" if gt < -0.02 else "-> neutre")
        print(f"    c{i} : gamma={g:+.4f}  tanh(gamma)={gt:+.4f}  ||W_coh||={w_coh:.2f}  {role}")
    print("-" * 78)
    print("  Viscosite par couche :")
    for i in range(meta["n_layers"]):
        print(f"    c{i} : nu_diff={stats.layer_nu_diff[i]:.4f}  "
              f"nu_adv={stats.layer_nu_adv[i]:.4f}")
    print("-" * 78)
    print("  Embeddings :")
    print(f"    Rang effectif   : {stats.emb_effective_rank}/{meta['embed_dim']}")
    print(f"    Entropie spect. : {stats.emb_spectral_entropy:.3f}/{math.log(meta['embed_dim']):.3f}")
    print(f"    Tete head rang effectif : {stats.head_effective_rank}/{meta['embed_dim']}")
    print("-" * 78)
    print("  Normes matrices (||W||) les plus grandes :")
    top = sorted(stats.matrix_norms.items(), key=lambda kv: kv[1], reverse=True)[:8]
    for name, val in top:
        rank = stats.matrix_ranks.get(name, -1)
        print(f"    {name:<24}  ||W||={val:6.2f}  rang eff={rank}")
    print("=" * 78)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", type=str, default="",
                    help="Chemin explicite du .pt (sinon, le plus récent)")
    args = ap.parse_args()

    ckpt_path = find_checkpoint(args.checkpoint)
    print(f"[INFO] Chargement : {ckpt_path}")
    model, tokenizer, meta = load_model_and_tokenizer(ckpt_path)
    stats = extract_stats(model)
    stats.meta = meta

    print(f"[INFO] Modèle : {meta['n_params']:,} params")
    print(f"[INFO] Config : D={meta['embed_dim']}  L={meta['seq_len']}  "
          f"n_layers={meta['n_layers']}  n_heads={meta['n_heads']}")
    print()
    print("[INFO] Génération des figures…")

    figures = [
        ("01 Architecture",              lambda: plot_01_architecture(meta)),
        ("02 γ par couche",              lambda: plot_02_gamma(stats)),
        ("03 Spectres SVD",              lambda: plot_03_weight_spectra(model, stats)),
        ("04 Entropie attention",        lambda: plot_04_attention_entropy(model, tokenizer, stats)),
        ("05 Normes latentes",           lambda: plot_05_latent_norms(model, tokenizer)),
        ("06 Biais global/token",        lambda: plot_06_global_bias_per_token(model, tokenizer)),
        ("07 Embeddings PCA+SVD",        lambda: plot_07_embedding_space(model, tokenizer, stats)),
        ("08 Viscosité",                 lambda: plot_08_viscosity(model, tokenizer, stats)),
        ("09 Distributions poids",       lambda: plot_09_weight_distributions(model)),
        ("10 Head weights",              lambda: plot_10_head_weights(model, stats)),
        ("11 Multi-head attention",      lambda: plot_11_multihead_attention(model, tokenizer)),
        ("12 Contributions par couche",  lambda: plot_12_layer_contributions(model, tokenizer)),
        ("13 Gate RoPE",                 lambda: plot_13_rope_gate(model, tokenizer)),
    ]

    for name, fn in figures:
        try:
            out = fn()
            print(f"  [OK]  {name:<32}  -> {Path(out).name}")
        except Exception as e:
            print(f"  [ERREUR]  {name}  :  {e!r}")

    # Export JSON brut
    stats_path = FIG_DIR / "stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        # Conversion safe (numpy → python)
        def safe(v):
            if isinstance(v, (np.floating, np.integer)): return v.item()
            if isinstance(v, np.ndarray): return v.tolist()
            return v
        d = stats.to_dict()
        out_json = {k: (safe(v) if not isinstance(v, (list, dict)) else v) for k, v in d.items()}
        json.dump(out_json, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[INFO] Stats brutes exportées : {stats_path.name}")

    print_text_report(meta, stats)
    print(f"\nToutes les figures sont dans : {FIG_DIR.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
