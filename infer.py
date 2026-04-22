#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from tokenizers import Tokenizer

from model.mira_glu_rope import MiraGLURopeLM


ROLE_EOT = "<|endofturn|>"


def load_bundle(checkpoint_path: Path, device: torch.device):
    ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or "state_dict" not in ck:
        raise ValueError("Checkpoint invalide (dict + state_dict attendu).")

    vocab_size = int(ck["vocab_size"])
    seq_len = int(ck["seq_len"])
    embed_dim = int(ck.get("embed_dim", 256))
    n_layers = int(ck.get("n_layers", 6))
    n_heads = int(ck.get("n_heads", 8))
    pre_ln = bool(ck.get("pre_ln", False))

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
    ).to(device)
    model.load_state_dict(ck["state_dict"], strict=True)
    model.eval()

    tok_path = ck.get("tokenizer_path", "tokenizer/tokenizer_v3_fr12k.json")
    tok_path = Path(str(tok_path))
    if not tok_path.is_absolute():
        tok_path = checkpoint_path.parent.parent / tok_path
    tokenizer = Tokenizer.from_file(str(tok_path))

    eot_id = None
    try:
        eot_id = tokenizer.token_to_id(ROLE_EOT)
    except Exception:
        eot_id = None

    return model, tokenizer, eot_id


@torch.no_grad()
def generate(
    model: MiraGLURopeLM,
    tokenizer: Tokenizer,
    prompt: str,
    device: torch.device,
    *,
    max_new: int,
    temperature: float,
    top_k: int,
    top_p: float,
    rep_penalty: float,
    eot_id: int | None,
) -> str:
    prompt_ids = tokenizer.encode(prompt).ids
    if not prompt_ids:
        return ""

    context = list(prompt_ids[-model.seq_len :])
    generated: list[int] = []

    for _ in range(int(max_new)):
        x = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(x)[0, -1, :]

        for tok in set(generated):
            logits[tok] = logits[tok] / rep_penalty if logits[tok] > 0 else logits[tok] * rep_penalty

        logits = logits / max(float(temperature), 1e-8)

        if int(top_k) > 0:
            kth = torch.topk(logits, min(int(top_k), logits.numel())).values[-1]
            logits = logits.masked_fill(logits < kth, float("-inf"))

        if float(top_p) > 0.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            probs_sorted = F.softmax(sorted_logits, dim=-1)
            cum = torch.cumsum(probs_sorted, dim=-1)
            remove = (cum - probs_sorted) > float(top_p)
            sorted_logits[remove] = float("-inf")
            logits = torch.zeros_like(logits).scatter_(0, sorted_idx, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        tok_id = int(torch.multinomial(probs, 1).item())

        if eot_id is not None and tok_id == int(eot_id):
            break

        generated.append(tok_id)
        context = (context + [tok_id])[-model.seq_len :]

    return tokenizer.decode(generated)


def main() -> None:
    ap = argparse.ArgumentParser(description="Inference MiraGLURopeLM (sampling)")
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoint/mira_glu_rope_v4_v3roles_ft_wiki02_ep3.pt"),
    )
    ap.add_argument(
        "--prompt",
        type=str,
        default="<|user|> Bonjour, comment vas-tu ? <|endofturn|>\n<|mira|>",
    )
    ap.add_argument("--max-new", type=int, default=250)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.92)
    ap.add_argument("--rep-penalty", type=float, default=1.25)
    ap.add_argument("--device", type=str, default="", help="cpu / cuda (defaut: auto)")
    args = ap.parse_args()

    dev = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = args.checkpoint
    if not ckpt.is_absolute():
        ckpt = Path(__file__).resolve().parent / ckpt
    if not ckpt.is_file():
        raise FileNotFoundError(str(ckpt))

    model, tok, eot_id = load_bundle(ckpt, dev)
    out = generate(
        model,
        tok,
        args.prompt,
        dev,
        max_new=int(args.max_new),
        temperature=float(args.temperature),
        top_k=int(args.top_k),
        top_p=float(args.top_p),
        rep_penalty=float(args.rep_penalty),
        eot_id=eot_id,
    )
    print(out.strip())


if __name__ == "__main__":
    main()

