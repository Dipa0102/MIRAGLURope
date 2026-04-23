# `MiraGLURopeLM` — model internals (English)

This document explains how the model implemented in `model/mira_glu_rope.py` works, focusing on how the following pieces connect:

- **multi-head attention** (the \(A(\cdot)\) branch),
- the **global coherence bias** (added to attention logits),
- **viscosity control** \( \nu \),
- the **bilinear FFN / GLU** branch with **advection**,
- and **RoPE** usage (hybrid RoPE in attention, dedicated RoPE in the GLU path).

`MiraGLURopeLM` is a causal language model (decoder-only): it consumes a token sequence and predicts the next token.

## High-level pipeline

In `MiraGLURopeLM.forward(token_ids)`:

1. **Embeddings**
   - `tok_emb(token_ids)`: token embedding
   - `pos_emb(pos)`: learned absolute position embedding
   - sum: \( z_0 = e_{\text{tok}} + e_{\text{pos}} \) with shape \((B, L, D)\)

2. **Model body**
   - `SequenceModelGLURope` stacks \(N\) `SequenceLayerGLURope` blocks
   - each block applies a residual update:

\[
z' = z + \alpha_{\text{attn}}\;A(z) + \alpha_{\text{mlp}}\; \text{MLP}_\text{adv}(z)
\]

3. **Output**
   - final `LayerNorm`
   - `head: Linear(D → V)` → logits \((B, L, V)\)

## Notation and shapes

- \(B\): batch size
- \(L\): sequence length (≤ `seq_len`)
- \(D\): embedding size (`embed_dim`)
- \(H\): number of heads (`n_heads`)
- \(d_h = D/H\): per-head dimension

## Inside one `SequenceLayerGLURope`

Each layer has **two branches**:

- **attention branch** `kernel_A`: computes \(A(z)\)
- **bilinear (GLU) + advection branch**: computes \(\text{MLP}_\text{adv}(z)\)

The layer combines them in a residual with learned/scalar gains:

- `alpha_attn` (default 0.42)
- `alpha_mlp` (default 1.07)
- `alpha_adv` (default 1.0), scaling the advection term

There are two normalization modes:

- **Legacy (LN only on the MLP branch output)**: `pre_ln=False`
- **Pre-LN**: `pre_ln=True` (LayerNorm before attention and before MLP, plus `residual_scale`)

### 1) Per-position causal pooling (causality-critical)

The layer constructs a **causal pool per position**:

\[
\text{pool\_seq}[t] = \frac{1}{t+1}\sum_{i=0}^{t} z[i]
\]

In code: `_causal_cummean(z)` returns \((B,L,D)\).  
This avoids causal leakage that would happen with a “global mean over the full window”.

This `pool_seq` is used in two places:

- to **parameterize hybrid RoPE** (a context-dependent gate)
- for the **global bias** (a context-dependent attention bias)

### 2) Multi-head attention with hybrid RoPE + global bias

`KernelCouplingA_CausalPool` performs:

#### a) Q/K/V projections

\[
Q = W_q z,\;\; K_{\text{raw}} = W_k z,\;\; V = W_v z
\]

Shapes: \((B,L,D)\).

#### b) Global coherence bias (added to attention logits)

`GlobalCoherenceBias` projects a **global/context vector** into a “global key”:

\[
g_k[t] = W_{\text{coh}}\;\text{pool\_seq}[t]
\]

Then it builds a bias against all keys \(K_{\text{raw}}\):

\[
\text{gb}[t, j] = \tanh(\gamma)\;\frac{\langle g_k[t],\, K_{\text{raw}}[j]\rangle}{\sqrt{D}}
\]

In the code, this is materialized as a \((B,L,L)\) matrix (query \(t\) × key \(j\)) and added to the attention logits.

**Intuition**: this injects a causal “global summary” signal that can attract/repel certain keys independently of the \(QK^\top\) match.

#### c) Hybrid RoPE in attention (fast/slow + gate)

The model applies positional rotations to **Q and K** via `apply_rope_hybrid`:

- “fast” base: `ROPE_BASE_FAST = φ` (golden ratio)
- “slow” base: `ROPE_BASE_SLOW = 1618`
- a learned **gate** controls the mix:

\[
\text{gate}[t] = \sigma(\text{MLP}(\text{pool\_seq}[t]))
\]

The RoPE cos/sin are interpolated:

\[
\cos = \text{gate}\cdot \cos_{\text{fast}} + (1-\text{gate})\cdot \cos_{\text{slow}}
\]
(same for \(\sin\)).

Then standard RoPE rotation (pairwise):

\[
\text{RoPE}(x)= x\odot \cos + \text{rotate\_half}(x)\odot \sin
\]

**Key point**: the gate depends on `pool_seq[t]`, so fast/slow preference can vary **per position** and **per prompt**.

#### d) Logits, causal mask, softmax

Reshape into heads:

- \(Q_h, K_h, V_h\): \((B,H,L,d_h)\)

Logits:

\[
\text{logits} = \frac{Q_h K_h^\top}{\sqrt{d_h}} + \text{gb}
\]

Apply a strict upper-triangular causal mask, then softmax over keys and aggregate:

\[
\text{out} = \text{softmax}(\text{logits})\;V
\]

The layer also returns `attn_w` (the attention matrix averaged across heads) for diagnostics.

### 3) `DuplexViscosity`: advection modulation

`DuplexViscosity(dim)` produces an **effective viscosity** \(\nu\ge 0\) from the pool (context):

- two learned scalars: `nu_diff` and `nu_adv` (used as absolute values)
- a small MLP `proj` that outputs a bounded modulation via `tanh`

In code:

\[
\nu = |\nu_{\text{diff}}| + \tanh(\text{proj}(\text{pool}))\cdot |\nu_{\text{adv}}|
\]

In `SequenceLayerGLURope`, \(\nu\) is evaluated **per position** and reshaped to \((B,L,1)\) to scale the advection term.

### 4) Bilinear FFN / GLU + advection (with dedicated RoPE)

This “MLP” is not a classic \(W_2 \sigma(W_1 z)\) feed-forward.  
Instead, it uses a **bilinear interaction** (GLU-like):

\[
\text{bilinear}(z) = W_u(z)\;\odot\;W_g(z)
\]

Then an advection-like term (note the negative sign):

\[
\text{adv}(z) = -\alpha_{\text{adv}}\;\nu\;\text{bilinear}(z)
\]

And finally an output projection `C` (with LN in legacy mode):

- **legacy** (`pre_ln=False`):

\[
\text{MLP}_\text{adv}(z) = C(\text{LN}(\text{adv}(z)))
\]

- **pre-LN** (`pre_ln=True`):

\[
\text{MLP}_\text{adv}(z) = C(\text{adv}(z))
\]
(normalization happens upstream via `ln2`).

#### RoPE on the GLU path (this model’s signature change)

Before computing \(W_u(\cdot)\) and \(W_g(\cdot)\), the layer applies a RoPE **dedicated** to the GLU input:

\[
z_{\text{glu}} = \text{RoPE}_{\text{glu}}(z)
\]

with a **slow** base by default:

- `rope_base_glu_slow = 16180` (constant `ROPE_BASE_GLU_DEFAULT`)

Then:

\[
\text{bilinear} = W_u(z_{\text{glu}})\odot W_g(z_{\text{glu}})
\]

**Key difference vs attention RoPE**:

- attention: **hybrid** RoPE (fast \(φ\) / slow 1618) + context-dependent gate
- GLU: **fixed** RoPE, single slow base (16180), no fast/slow mixing, no gate

**Intuition**: advection depends on a bilinear product; using a very slow RoPE in that path yields a gentle position-dependent variation, consistent with a “transport” signal rather than fast matching as in attention.

## `pre_ln=True` variant (Pre-LayerNorm + scaling)

When `pre_ln=True`, a layer does:

1. `z1 = ln1(z)` then attention on `z1`
2. residual update: \(z \leftarrow z + (\text{residual\_scale}\cdot \alpha_{\text{attn}})\;A(z1)\)
3. `z2 = ln2(z)` then GLU/advection branch on `z2`
4. residual update: \(z \leftarrow z + (\text{residual\_scale}\cdot \alpha_{\text{mlp}})\;\text{MLP}_\text{adv}(z2)\)

Default `residual_scale`:

\[
\text{residual\_scale} = \frac{1}{\sqrt{2N}}
\]

where \(N\) is the number of layers; this helps keep residual additions stable.

## How “attention heads” fit together (multi-head)

The model follows standard multi-head attention:

- project Q/K/V in \(D\)
- split into \(H\) heads of size \(d_h\)
- each head computes its own \((L\times L)\) attention matrix
- head outputs are concatenated back to \(D\)

In `KernelCouplingA_CausalPool`, the returned `out` is already \((B,L,D)\).  
For diagnostics, `attn_w` is the **mean attention weights across heads**.

## Important note: learned absolute positions + RoPE

The model adds **learned absolute position embeddings** (`pos_emb`) and also applies **RoPE** in:

- attention (on Q/K),
- the GLU path (on the input of the bilinear interaction).

So positional information is represented by two mechanisms:

- **learned absolute positions** (pos_emb),
- **relative rotational positions** (RoPE).

## Where each component lives in the code

- Model: `model/mira_glu_rope.py`
  - `MiraGLURopeLM`: embeddings + stack + head
  - `SequenceModelGLURope`: layer stacking
  - `SequenceLayerGLURope`: attention + GLU/advection logic
  - `KernelCouplingA_CausalPool`: attention (QKV, hybrid RoPE, global bias, causal mask)
  - `GlobalCoherenceBias`: global bias (parameter `gamma`)
  - `DuplexViscosity`: viscosity \(\nu\)
  - `_causal_cummean`: per-position causal pooling

## Appendix: compact equation (legacy mode)

The file docstring summarizes the intent as:

\[
z' = z + \alpha_{\text{attn}}\;A(z) + \alpha_{\text{mlp}}\;C(\text{LN}(-\alpha_{\text{adv}}\;\nu\;(W_u\odot W_g)(z)))
\]

with this variant’s specific change:

1. compute \(z_{\text{glu}} = \text{RoPE}_{\text{glu}}(z)\)
2. replace \((W_u\odot W_g)(z)\) by \((W_u\odot W_g)(z_{\text{glu}})\)

