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


# ---------------- Separator/content detection (robust) ----------------
_ALNUM_RE = re.compile(r"[A-Za-z0-9]")

def normalize_for_content_check(s: str) -> str:
    """
    Normalizza stringa decodificata di un singolo token per capire se è "contenuto".
    Rimuove marker comuni e strip.
    """
    s = "" if s is None else str(s)
    # marker comuni (può capitare nei token raw o in alcuni decode)
    s = s.replace("▁", " ")
    s = s.replace("Ġ", " ")
    s = s.replace("Ċ", "\n")
    s = s.strip()
    return s

def is_content_token(decoded_single_token: str) -> bool:
    """
    True se, dopo normalizzazione, contiene almeno un alfanumerico.
    Questo include numeri e parole, e esclude whitespace/punteggiatura pura.
    """
    t = normalize_for_content_check(decoded_single_token)
    return bool(_ALNUM_RE.search(t))

def find_next_content_token(
    tokenizer,
    input_ids_row: torch.Tensor,   # [seq_len]
    start_pos: int,
    real_len: int,
    max_ahead: int = 400,
) -> Tuple[Optional[int], Optional[str]]:
    """
    Trova il primo indice j > start_pos tale che token[j] è "contenuto".
    Ritorna (j, decoded_str) o (None, None).
    """
    end = min(real_len, start_pos + 1 + max_ahead)
    for j in range(start_pos + 1, end):
        tid = int(input_ids_row[j].item())
        s = tokenizer.decode([tid], clean_up_tokenization_spaces=False)
        if is_content_token(s):
            return j, s
    return None, None


# ---------------- Main ----------------
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

    ap.add_argument("--token-step", type=int, default=100)
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--save-ground-truth-next-n", type=int, default=0)
    ap.add_argument("--max-ahead", type=int, default=400)
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
                "token_step": int(args.token_step),
                "topk": int(args.topk),
                "max_ahead": int(args.max_ahead),
                "filtering": "anchor must be CONTENT; target is NEXT CONTENT token (skip whitespace/punctuation)",
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
                        torch.save(emb_obj, emb_path)

                # ---- top-k NTP: anchor content -> target next content ----
                step = int(args.token_step)
                if step > 0 and int(args.topk) > 0:
                    for b in range(input_ids_all.shape[0]):
                        real_len = int(attn_mask_all[b].sum().item()) if attn_mask_all is not None else input_ids_all.shape[1]
                        if real_len <= 2:
                            continue

                        line_index = global_line_offset + b
                        positions = list(range(step - 1, real_len, step))
                        if not positions:
                            continue

                        kept = 0
                        skipped_anchor_not_content = 0
                        skipped_no_target = 0

                        for pos in positions:
                            if pos >= real_len - 1:
                                continue

                            anchor_id = int(input_ids_all[b, pos].item())
                            anchor_str = tokenizer.decode([anchor_id], clean_up_tokenization_spaces=False)

                            # anchor deve essere contenuto
                            if not is_content_token(anchor_str):
                                skipped_anchor_not_content += 1
                                continue

                            j, target_str = find_next_content_token(
                                tokenizer=tokenizer,
                                input_ids_row=input_ids_all[b],
                                start_pos=pos,
                                real_len=real_len,
                                max_ahead=int(args.max_ahead),
                            )
                            if j is None:
                                skipped_no_target += 1
                                continue

                            pred_pos = j - 1
                            if pred_pos < 0:
                                skipped_no_target += 1
                                continue

                            tk = topk_from_logits(tokenizer, logits[b, pred_pos, :], topk=int(args.topk))
                            tgt_id = int(input_ids_all[b, j].item())

                            rec: Dict[str, Any] = {
                                "line_index": int(line_index),

                                "anchor_pos_0based": int(pos),
                                "anchor_pos_1based": int(pos + 1),
                                "anchor_token_id": int(anchor_id),
                                "anchor_token_str": anchor_str,

                                "target_pos_0based": int(j),
                                "target_pos_1based": int(j + 1),
                                "true_target_token_id": int(tgt_id),
                                "true_target_token_str": target_str,

                                "pred_pos_0based": int(pred_pos),
                                "pred_pos_1based": int(pred_pos + 1),

                                **tk,
                            }

                            n_gt = int(args.save_ground_truth_next_n)
                            if n_gt > 0:
                                gt_start = pred_pos + 1
                                gt_end = min(real_len, pred_pos + 1 + n_gt)
                                gt_ids = input_ids_all[b, gt_start:gt_end].detach().cpu().tolist()
                                rec["ground_truth_next_token_ids"] = gt_ids
                                rec["ground_truth_next_tokens"] = tokenizer.convert_ids_to_tokens(gt_ids)
                                rec["ground_truth_next_decoded"] = decode_token_list(tokenizer, gt_ids)

                            out_obj["records"].append(rec)
                            kept += 1

                        print(
                            f"[TOPK] line_index={line_index} candidates={len(positions)} kept={kept} "
                            f"skipped_anchor_not_content={skipped_anchor_not_content} skipped_no_target={skipped_no_target}"
                        )

                        if args.save_skip_stats:
                            out_obj["skip_stats_by_line"].append(
                                {
                                    "line_index": int(line_index),
                                    "candidates": int(len(positions)),
                                    "kept": int(kept),
                                    "skipped_anchor_not_content": int(skipped_anchor_not_content),
                                    "skipped_no_target": int(skipped_no_target),
                                }
                            )

                global_line_offset += len(chunk)

            out_path = out_dir / f"final_topk_every{args.token_step}tok_{input_file_name}.pt"
            if not (args.skip_if_exists and out_path.exists()):
                torch.save(out_obj, out_path)
                print(f"[OK] Top-k salvato: {out_path}")
            else:
                print(f"[SKIP] esiste già: {out_path}")

            print(f"[OK] Embeddings salvati per: {fpath}")

    print("\nDone.")


if __name__ == "__main__":
    main()