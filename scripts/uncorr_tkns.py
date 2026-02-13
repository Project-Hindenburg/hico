#!/usr/bin/env python3
"""Pick maximally different single-token lowercase words from english.txt.

Method:
1) Keep lowercase alphabetic words only.
2) Keep words that tokenize to exactly one token for the exact prompt surface form " " + word.
3) Embed each candidate token at a user-chosen model layer.
4) Greedy farthest-point sampling with cosine distance.
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

WORD_RE = re.compile(r"^[a-z]+$")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", type=str, default="meta-llama/Llama-3.1-8B")
    p.add_argument("--wordlist", type=Path, default=Path("english.txt"))
    p.add_argument("--layer", type=int, default=0, help="0=embeddings, 1..N=transformer layers")
    p.add_argument("--k", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", type=Path, default=Path("selected_tokens.txt"))
    return p.parse_args()


def load_words(path: Path) -> list[str]:
    words: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            w = line.strip().lower()
            if WORD_RE.fullmatch(w):
                words.append(w)
    return words


def build_candidates(words: list[str], tokenizer) -> list[tuple[str, int, str]]:
    candidates: list[tuple[str, int, str]] = []
    seen_ids: set[int] = set()

    for w in words:
        surface = " " + w
        ids = tokenizer(surface, add_special_tokens=False).input_ids
        if len(ids) != 1:
            continue
        tid = int(ids[0])
        if tid in seen_ids:
            continue
        seen_ids.add(tid)
        candidates.append((w, tid, surface))

    return candidates


def normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


@torch.inference_mode()
def embed_at_layer(
    model,
    tokenizer,
    token_ids: list[int],
    layer: int,
    batch_size: int,
    device: str,
) -> np.ndarray:
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else model.config.bos_token_id
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else model.config.eos_token_id

    vectors: list[np.ndarray] = []

    for start in range(0, len(token_ids), batch_size):
        chunk = token_ids[start : start + batch_size]
        seqs: list[list[int]] = []
        token_positions: list[int] = []
        for tid in chunk:
            if cls_id is not None and sep_id is not None:
                seqs.append([int(cls_id), tid, int(sep_id)])
                token_positions.append(1)
            elif bos_id is not None:
                seqs.append([int(bos_id), tid])
                token_positions.append(1)
            elif eos_id is not None:
                seqs.append([int(eos_id), tid])
                token_positions.append(1)
            else:
                seqs.append([tid])
                token_positions.append(0)

        input_ids = torch.tensor(seqs, dtype=torch.long, device=device)

        out = model(input_ids=input_ids, output_hidden_states=True)
        hs = out.hidden_states

        layer_idx = layer if layer >= 0 else len(hs) + layer
        if layer_idx < 0 or layer_idx >= len(hs):
            raise ValueError(
                f"Invalid --layer={layer}. Valid range is [{-(len(hs))}, {len(hs)-1}] for this model."
            )

        row_idx = torch.arange(len(chunk), device=device)
        col_idx = torch.tensor(token_positions, device=device)
        vec = hs[layer_idx][row_idx, col_idx, :].float().cpu().numpy()
        vectors.append(vec)

    x = np.concatenate(vectors, axis=0)
    return normalize_rows(x)


def farthest_point_sampling(vecs: np.ndarray, k: int, seed: int) -> list[int]:
    n = vecs.shape[0]
    if n == 0:
        return []

    rng = random.Random(seed)
    chosen = [rng.randrange(n)]

    max_sim = vecs @ vecs[chosen[0]]
    max_sim[chosen[0]] = np.inf

    while len(chosen) < min(k, n):
        idx = int(np.argmin(max_sim))
        chosen.append(idx)
        max_sim = np.maximum(max_sim, vecs @ vecs[idx])
        max_sim[chosen] = np.inf

    return chosen


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModel.from_pretrained(
        args.model_id,
        dtype=torch.float16 if args.device.startswith("cuda") else torch.float32,
    ).to(args.device)
    model.eval()

    words = load_words(args.wordlist)
    candidates = build_candidates(words, tokenizer)

    if len(candidates) < args.k:
        raise RuntimeError(f"Only {len(candidates)} candidate single-token words found, need at least {args.k}.")

    token_ids = [tid for _, tid, _ in candidates]
    vecs = embed_at_layer(model, tokenizer, token_ids, args.layer, args.batch_size, args.device)
    selected_idx = farthest_point_sampling(vecs, args.k, args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for i in selected_idx:
            word, tid, surface = candidates[i]
            f.write(f"{word}\t{tid}\t{surface}\n")

    print(f"candidates={len(candidates)}")
    print(f"selected={len(selected_idx)}")
    print(f"saved={args.output.resolve()}")


if __name__ == "__main__":
    main()



"""
python uncorr_tkns.py --model-id meta-llama/Llama-3.1-8B --wordlist data/english.txt --layer 26 --k 64 --device cuda --output outputdir/selected_llama31_layer26.txt
"""