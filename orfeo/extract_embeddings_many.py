#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import Iterable, List, Dict, Any, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------- IO helpers ----------------
def iter_input_files(input_root: Path, patterns: List[str]) -> Iterable[Path]:
    for pat in patterns:
        yield from input_root.rglob(pat)


def read_nonempty_lines(p: Path) -> List[str]:
    lines = p.read_text(encoding="utf-8").splitlines()
    lines = [ln.rstrip("\n") for ln in lines]
    lines = [ln for ln in lines if ln.strip() != ""]
    return lines


def batched(lst: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]


def decode_token_list(tokenizer, ids: List[int]) -> List[str]:
    return [tokenizer.decode([tid], clean_up_tokenization_spaces=False) for tid in ids]


# ---------------- Top-k from final logits ----------------
def topk_from_logits(tokenizer, logits_vec: torch.Tensor, topk: int) -> Dict[str, Any]:
    probs = torch.softmax(logits_vec.float(), dim=-1)
    k = min(int(topk), int(probs.shape[-1]))
    top_probs, top_ids = torch.topk(probs, k=k, dim=-1)

    top_ids_list = top_ids.detach().cpu().tolist()
    top_probs_list = top_probs.detach().cpu().tolist()
    top_tokens = tokenizer.convert_ids_to_tokens(top_ids_list)
    top_decoded = [tokenizer.decode([tid], clean_up_tokenization_spaces=False) for tid in top_ids_list]

    return {
        "top_token_ids": top_ids_list,
        "top_tokens": top_tokens,
        "top_decoded": top_decoded,
        "top_probs": top_probs_list,
    }


# ---------------- Token classification ----------------
_PUNCT_ONLY = re.compile(r'^[,.;:!?()\[\]{}\-–—/\\\'"`]+$')

def norm_token_text(s: str) -> str:
    s = "" if s is None else str(s)
    s = s.replace("▁", " ").replace("Ġ", " ").replace("Ċ", "\n")
    return s.strip()

def is_element_token(decoded_single_token: str) -> bool:
    """
    True for elements (numbers or words). False for separators (whitespace/punct-only).
    Works for both:
      - numbers: "14"
      - words: " Monday" -> "Monday"
    """
    t = norm_token_text(decoded_single_token)
    if t == "" or t.startswith("<|"):
        return False
    if _PUNCT_ONLY.fullmatch(t):
        return False
    return True


def collect_element_positions(tokenizer, input_ids_row: torch.Tensor, real_len: int) -> List[int]:
    """
    Return list of token positions whose decoded single-token is an element (not separator).
    """
    pos = []
    for j in range(real_len):
        s = tokenizer.decode([int(input_ids_row[j].item())], clean_up_tokenization_spaces=False)
        if is_element_token(s):
            pos.append(j)
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)

    ap.add_argument("--filelist", type=Path, default=None)
    ap.add_argument("--patterns", nargs="+", default=["*_2000.txt"])
    ap.add_argument("--layer-num", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--skip-if-exists", action="store_true")
    ap.add_argument("--save-embedding",type=bool, default=False)

    # NEW: sample every N pairs (A,B) instead of every N tokenizer tokens
    ap.add_argument("--pair-step", type=int, default=50, help="Salva un record ogni N coppie (A,B).")
    ap.add_argument("--topk", type=int, default=200)

    ap.add_argument("--save-ground-truth-next-n", type=int, default=0)
    ap.add_argument("--save-skip-stats", action="store_true")

    args = ap.parse_args()

    if args.filelist is not None:
        files = [Path(x.strip()) for x in args.filelist.read_text(encoding="utf-8").splitlines() if x.strip()]
    else:
        files = sorted(set(iter_input_files(args.input_root, args.patterns)))

    if not files:
        raise ValueError("Nessun file trovato (controlla input_root/patterns o filelist).")

    print(f"Trovati {len(files)} file input.")

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir), use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir),
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    device = next(model.parameters()).device
    print(f"Device principale: {device}")

    args.output_root.mkdir(parents=True, exist_ok=True)

    with torch.inference_mode():
        for fpath in files:
            try:
                rel_dir = fpath.parent.relative_to(args.input_root)
            except ValueError:
                rel_dir = Path()

            out_dir = args.output_root / rel_dir
            out_dir.mkdir(parents=True, exist_ok=True)

            lines = read_nonempty_lines(fpath)
            if not lines:
                print(f"[SKIP] {fpath} (only empty lines)")
                continue

            input_file_name = fpath.stem
            print(f"\n=== FILE: {fpath} | non empty lines: {len(lines)} | OUT: {out_dir} ===")

            out_obj: Dict[str, Any] = {
                "source_file": str(fpath),
                "layer_num": int(args.layer_num),
                "pair_step": int(args.pair_step),
                "topk": int(args.topk),
                "mode": "PAIR_AB (intra-pair A->B), sampled in pair-space",
                "records": [],
            }
            if args.save_skip_stats:
                out_obj["skip_stats_by_line"] = []

            global_line_offset = 0

            for chunk in batched(lines, batch_size=int(args.batch_size)):
                inputs = tokenizer(
                    chunk,
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                    return_attention_mask=True,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                for b in range(inputs["input_ids"].shape[0]):
                    real_len_b = int(inputs["attention_mask"][b].sum().item())
                    print(f"[TOKENS] file={fpath.name} line_index={global_line_offset+b} real_len={real_len_b}")

                outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                hidden_states = outputs.hidden_states
                logits = outputs.logits  # [B, seq, vocab]

                if args.layer_num < 0 or args.layer_num >= len(hidden_states):
                    raise ValueError(
                        f"layer_num={args.layer_num} fuori range. hidden_states disponibili: 0..{len(hidden_states)-1}"
                    )

                layer_reps = hidden_states[int(args.layer_num)]
                input_ids_all = inputs["input_ids"]
                attn_mask_all = inputs.get("attention_mask", None)

                # ---- embeddings ----
                for b in range(layer_reps.shape[0]):
                    real_len = int(attn_mask_all[b].sum().item()) if attn_mask_all is not None else layer_reps.shape[1]
                    if real_len <= 0:
                        continue

                    reps_seq = layer_reps[b, :real_len].detach().cpu().to(torch.float16)
                    input_ids_seq = input_ids_all[b, :real_len].detach().cpu()

                    line_index = global_line_offset + b
                    emb_obj = {
                        "input_ids": input_ids_seq,
                        "embeddings": reps_seq,
                        "line_index": int(line_index),
                        "source_file": str(fpath),
                        "layer_num": int(args.layer_num),
                    }

                    emb_path = out_dir / f"reprs_{input_file_name}_line{line_index:06d}_layer{args.layer_num}.pt"
                    if not (args.skip_if_exists and emb_path.exists()):
                        if args.save_embedding:
                            torch.save(emb_obj, emb_path)

                # ---- records: save ONLY previous element + logits that predict the NEXT element ----
                for b in range(input_ids_all.shape[0]):
                    real_len = int(attn_mask_all[b].sum().item()) if attn_mask_all is not None else input_ids_all.shape[1]
                    if real_len <= 1:
                        continue

                    line_index = global_line_offset + b

                    elem_pos = collect_element_positions(tokenizer, input_ids_all[b], real_len)
                    if len(elem_pos) < 2:
                        continue  # need at least a prev and a next element

                    pair_step = max(1, int(args.pair_step))
                    kept = 0

                    # iterate over consecutive element pairs (prev -> next)
                    # sample every N pairs in element-space
                    for ei in range(0, len(elem_pos) - 1, pair_step):
                        prev_pos = elem_pos[ei]
                        next_pos = elem_pos[ei + 1]

                        # logits at (next_pos - 1) predict the token at next_pos
                        if next_pos <= 0:
                            continue
                        pred_pos = next_pos - 1
                        if pred_pos >= real_len:
                            continue

                        # previous element token (THIS is what you want to save)
                        prev_id = int(input_ids_all[b, prev_pos].item())
                        prev_raw = tokenizer.decode([prev_id], clean_up_tokenization_spaces=False)
                        prev_str = norm_token_text(prev_raw)

                        # (optional but often useful) what token is immediately before the predicted next number?
                        # This is typically whitespace/comma, helps you verify you're in the right place.
                        ctx_id = int(input_ids_all[b, pred_pos].item())
                        ctx_raw = tokenizer.decode([ctx_id], clean_up_tokenization_spaces=False)
                        ctx_str = norm_token_text(ctx_raw)

                        tk = topk_from_logits(tokenizer, logits[b, pred_pos, :], topk=int(args.topk))

                        rec = {
                            "line_index": int(line_index),

                            # the "current" number node
                            "anchor_pos_0based": int(prev_pos),
                            "anchor_pos_1based": int(prev_pos + 1),
                            "anchor_token_id": int(prev_id),
                            "anchor_token_str": prev_str,

                            # where the prediction distribution comes from
                            "pred_pos_0based": int(pred_pos),
                            "pred_pos_1based": int(pred_pos + 1),

                            "anchor_index_in_elements": int(ei),  # which number token is this in the line? (0-based)

                            # distribution for the next token (which should be the next number token)
                            **tk,
                        }

                        out_obj["records"].append(rec)
                        kept += 1

                    print(f"[PREV_ONLY] line_index={line_index} element_pairs={len(elem_pos)-1} kept={kept}")

                global_line_offset += len(chunk)

                # ---- Save final .pt file ----
                out_path = out_dir / f"final_topk_pairAB_step{args.pair_step}_{input_file_name}.pt"
                if not (args.skip_if_exists and out_path.exists()):
                    torch.save(out_obj, out_path)
                    print(f"[OK] Salvato: {out_path}")
                else:
                    print(f"[SKIP] esiste già: {out_path}")

                print(f"[OK] Embeddings salvati per: {fpath}")

if __name__ == "__main__":
    main()