# MIRA-GLURope

`MiraGLURopeLM` is a causal language model: token embeddings, stacked transformer-style blocks, final layer norm, and a vocabulary head. The implementation lives in a single module for easy inspection and reuse.

## Code layout

| Symbol | Role |
|--------|------|
| `MiraGLURopeLM` | Top-level model: `tok_emb`, `pos_emb`, `SequenceModelGLURope` body, `head` |
| `SequenceModelGLURope` | Stack of `SequenceLayerGLURope` with optional inter-layer dropout |
| `SequenceLayerGLURope` | Residual block: multi-head attention (`kernel_A`) + bilinear GLU branch with advection |
| `KernelCouplingA_CausalPool` | Q/K/V projections, hybrid RoPE on Q/K, global coherence bias, causal softmax attention |
| `GlobalCoherenceBias` | Scales a learned linear map on the pooled state into an additive key-side bias |
| `DuplexViscosity` | Produces a non-negative `ν` used to scale the advection term |
| `apply_rope_hybrid` | Interpolates fast/slow RoPE tables using a learned gate on the per-position pool |

**Module path:** `model/mira_glu_rope.py`  
**Public exports:** `MiraGLURopeLM`, `ROPE_BASE_GLU_DEFAULT`

**Default RoPE-related constants in code**

- Attention: `ROPE_BASE_FAST` (φ), `ROPE_BASE_SLOW` (1618)  
- GLU input path: `rope_base_glu_slow` default 16180 (`ROPE_BASE_GLU_DEFAULT`)

## Dependencies

For the model class only, `torch` is required. A full `requirements.txt` is provided if you use the same environment as the rest of the repository.

```bash
pip install torch
```

## Constructing the model

`MiraGLURopeLM` arguments (see `__init__` in `model/mira_glu_rope.py` for the full list):

- `vocab_size`, `seq_len` — `seq_len` sets the `pos_emb` table length and is the expected maximum index range for position embeddings.  
- `embed_dim`, `n_layers`, `n_heads` — `embed_dim` must be divisible by `n_heads`.  
- `alpha_attn`, `alpha_mlp`, `alpha_adv` — residual weights for the attention output, the MLP/GLU output, and the advection term inside the GLU path.  
- `pre_ln` — if `True`, uses pre-LN blocks and `residual_scale` (default \(1/\sqrt{2L}\) inside `SequenceModelGLURope`).  
- `inter_layer_dropout` — dropout between layers (not inside a single block).  
- `rope_base_fast`, `rope_base_slow` — attention hybrid RoPE bases.  
- `rope_base_glu_fast`, `rope_base_glu_slow` — the GLU branch uses the slow base (16180 by default); `rope_base_glu_fast` is kept for API symmetry with training checkpoints.

## Forward pass

- **Input:** `token_ids` — `LongTensor` of shape `(B, L)` with token indices.  
- **Output:** `logits` — `FloatTensor` of shape `(B, L, vocab_size)`.

The block update is: residual attention term plus residual MLP/GLU term, where the MLP/GLU path applies RoPE to the GLU input, forms a bilinear product, multiplies by \(-\alpha_{\text{adv}} \nu\), then `LayerNorm` (in the legacy path) and `C`.

## Minimal example

```python
import torch
from model.mira_glu_rope import MiraGLURopeLM

m = MiraGLURopeLM(
    vocab_size=8000,
    seq_len=128,
    embed_dim=128,
    n_layers=2,
    n_heads=2,
).eval()

x = torch.randint(0, 8000, (2, 64))
with torch.no_grad():
    logits = m(x)  # (2, 64, 8000)
```

## Further detail

Step-by-step math and tensor shapes: `docs/mira_glu_rope_model_explained_en.md`
