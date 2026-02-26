import argparse
from pathlib import Path
from typing import Iterable, List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


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


def apply_final_norm_if_available(model, hidden: torch.Tensor) -> torch.Tensor:
    """
    Applica la final norm del backbone se presente.
    hidden: [hidden_size] oppure [B, hidden_size]
    """
    backbone = getattr(model, "model", None)

    if backbone is not None:
        # LLaMA/Mistral-like
        if hasattr(backbone, "norm") and backbone.norm is not None:
            return backbone.norm(hidden)
        # GPTNeoX-like
        if hasattr(backbone, "final_layer_norm") and backbone.final_layer_norm is not None:
            return backbone.final_layer_norm(hidden)
        # OPT-like
        if hasattr(backbone, "decoder") and hasattr(backbone.decoder, "final_layer_norm"):
            fln = backbone.decoder.final_layer_norm
            if fln is not None:
                return fln(hidden)

    return hidden


def logit_lens_topk_for_hidden(
    model,
    tokenizer,
    hidden_last_token: torch.Tensor,   # [hidden_size]
    topk: int = 20,
) -> Dict[str, Any]:
    """
    Proietta hidden state intermedio su vocab (logit lens), softmax, top-k token.
    """
    hidden_last_token = hidden_last_token.to(next(model.parameters()).device)
    hidden_last_token = apply_final_norm_if_available(model, hidden_last_token)

    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise RuntimeError("Il modello non espone output embeddings / lm_head.")

    logits = lm_head(hidden_last_token)  # [vocab_size]
    probs = torch.softmax(logits.float(), dim=-1)

    k = min(topk, probs.shape[-1])
    top_probs, top_ids = torch.topk(probs, k=k, dim=-1)

    top_ids_list = top_ids.detach().cpu().tolist()
    top_probs_list = top_probs.detach().cpu().tolist()

    top_tokens = tokenizer.convert_ids_to_tokens(top_ids_list)
    top_texts = [tokenizer.decode([tid], clean_up_tokenization_spaces=False) for tid in top_ids_list]

    return {
        "top_token_ids": top_ids_list,
        "top_tokens": top_tokens,     # token grezzi (BPE/SPM)
        "top_decoded": top_texts,     # decode leggibile del singolo token
        "top_probs": top_probs_list,
    }


def decode_token_list(tokenizer, ids: List[int]) -> List[str]:
    return [tokenizer.decode([tid], clean_up_tokenization_spaces=False) for tid in ids]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)

    ap.add_argument("--filelist", type=Path, default=None)
    ap.add_argument("--patterns", nargs="+", default=["*_2000.txt"])
    ap.add_argument("--layer-num", type=int, default=26)
    ap.add_argument("--batch-size", type=int, default=1)  # consigliato 1 per prompt lunghi singoli
    ap.add_argument("--skip-if-exists", action="store_true")

    # Logit lens ogni k TOKEN (non batch)
    ap.add_argument(
        "--logit-lens-token-step",
        type=int,
        default=100,
        help="Calcola logit lens ogni k token lungo la sequenza (es. 100 => token #100, #200, ...).",
    )
    ap.add_argument(
        "--logit-lens-topk",
        type=int,
        default=20,
        help="Numero di token top-k da salvare come next-token prediction.",
    )
    ap.add_argument(
        "--save-ground-truth-next-n",
        type=int,
        default=20,
        help="Salva anche i prossimi N token reali del prompt dopo la posizione analizzata (0 per disabilitare).",
    )

    args = ap.parse_args()

    model_dir: Path = args.model_dir
    input_root: Path = args.input_root
    output_root: Path = args.output_root
    layer_num: int = args.layer_num
    batch_size: int = args.batch_size

    if args.filelist is not None:
        files = [Path(x.strip()) for x in args.filelist.read_text(encoding="utf-8").splitlines() if x.strip()]
    else:
        files = sorted(set(iter_input_files(input_root, args.patterns)))

    if not files:
        raise ValueError("Nessun file trovato (controlla input_root/patterns o filelist).")

    print(f"Trovati {len(files)} file input.")

    
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    device = next(model.parameters()).device
    print(f"Device principale: {device}")

    with torch.inference_mode():
        for fpath in files:
            try:
                rel_dir = fpath.parent.relative_to(input_root)
            except ValueError:
                rel_dir = Path()

            out_dir = output_root / rel_dir
            out_dir.mkdir(parents=True, exist_ok=True)

            lines = read_nonempty_lines(fpath)
            if not lines:
                print(f"[SKIP] {fpath} (only empty lines)")
                continue

            input_file_name = fpath.stem
            print(f"\n=== FILE: {fpath} | non empty lines: {len(lines)} | OUT: {out_dir} ===")

            # risultati logit lens per questo file
            logit_lens_results: Dict[str, Any] = {
                "source_file": str(fpath),
                "layer_num": layer_num,
                "token_step": args.logit_lens_token_step,
                "topk": args.logit_lens_topk,
                "save_ground_truth_next_n": args.save_ground_truth_next_n,
                "records": [],
            }

            global_line_offset = 0

            for chunk in batched(lines, batch_size=batch_size):
                inputs = tokenizer(
                    chunk,
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                    return_attention_mask=True,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                for b in range(inputs["input_ids"].shape[0]):
                    real_len = int(inputs["attention_mask"][b].sum().item())
                    print(f"[TOKENS] file={fpath.name} line_index={global_line_offset+b} real_len={real_len}")

                if real_len > 3000:
                    print(f"[LONG] {fpath} line={global_line_offset+b} tokens={real_len}")

                outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                hidden_states = outputs.hidden_states

                if layer_num < 0 or layer_num >= len(hidden_states):
                    raise ValueError(
                        f"layer_num={layer_num} fuori range. hidden_states disponibili: 0..{len(hidden_states)-1}"
                    )

                layer_reps = hidden_states[layer_num]  # [batch, seq_len, hidden]
                input_ids_all = inputs["input_ids"]
                attn_mask_all = inputs.get("attention_mask", None)

                # --- Salvataggio embeddings per ogni riga (come prima) ---
                for b in range(layer_reps.shape[0]):
                    if attn_mask_all is None:
                        real_len = layer_reps.shape[1]
                    else:
                        real_len = int(attn_mask_all[b].sum().item())

                    if real_len <= 0:
                        continue

                    start = 0
                    end = real_len

                    reps_seq = layer_reps[b, start:end].detach().to("cpu").to(torch.float16)
                    input_ids_seq = input_ids_all[b, start:end].detach().to("cpu")

                    line_index = global_line_offset + b

                    save_obj = {
                        "input_ids_last": input_ids_seq,
                        "embeddings_last": reps_seq,
                        "line_index": line_index,
                        "source_file": str(fpath),
                        "layer_num": layer_num,
                    }

                    out_path = out_dir / f"reprs_{input_file_name}_line{line_index:06d}_layer{layer_num}.pt"

                    if not (args.skip_if_exists and out_path.exists()):
                        torch.save(save_obj, out_path)

                # --- Logit lens ogni k TOKEN per ogni sequenza del batch ---
                step = args.logit_lens_token_step
                if step > 0:
                    for b in range(layer_reps.shape[0]):
                        if attn_mask_all is None:
                            real_len = layer_reps.shape[1]
                        else:
                            real_len = int(attn_mask_all[b].sum().item())

                        if real_len <= 0:
                            continue

                        line_index = global_line_offset + b

                        # posizioni 0-based: 99, 199, 299, ...
                        positions = list(range(step - 1, real_len, step))
                        if len(positions) == 0:
                            continue

                        # utile per debug
                        print(f"[LOGIT LENS] line_index={line_index} real_len={real_len} positions={positions[:5]}{'...' if len(positions)>5 else ''}")

                        for pos in positions:
                            hidden_tok = layer_reps[b, pos, :]  # [hidden_size]

                            topk_info = logit_lens_topk_for_hidden(
                                model=model,
                                tokenizer=tokenizer,
                                hidden_last_token=hidden_tok,
                                topk=args.logit_lens_topk,
                            )

                            input_tok_id = int(input_ids_all[b, pos].item())
                            input_tok_str = tokenizer.decode(
                                [input_tok_id],
                                clean_up_tokenization_spaces=False
                            )

                            rec: Dict[str, Any] = {
                                "line_index": line_index,
                                "token_position_0based": int(pos),
                                "token_position_1based": int(pos + 1),
                                "input_token_id": input_tok_id,
                                "input_token_str": input_tok_str,
                                **topk_info,
                            }

                            # (Opzionale) salva i prossimi N token reali del prompt (ground truth)
                            n_gt = args.save_ground_truth_next_n
                            if n_gt and n_gt > 0:
                                gt_start = pos + 1
                                gt_end = min(real_len, pos + 1 + n_gt)
                                gt_ids = input_ids_all[b, gt_start:gt_end].detach().to("cpu").tolist()
                                rec["ground_truth_next_token_ids"] = gt_ids
                                rec["ground_truth_next_tokens"] = tokenizer.convert_ids_to_tokens(gt_ids)
                                rec["ground_truth_next_decoded"] = decode_token_list(tokenizer, gt_ids)

                            logit_lens_results["records"].append(rec)

                global_line_offset += len(chunk)

            # Salva file separato con risultati logit lens per questo file input
            logit_out_path = out_dir / (
                f"logit_lens_every{args.logit_lens_token_step}tok_"
                f"{input_file_name}_layer{layer_num}.pt"
            )
            torch.save(logit_lens_results, logit_out_path)
            print(f"[OK] Logit lens salvato: {logit_out_path}")

            print(f"[OK] Embeddings saved for: {fpath}")

    print("\nDone.")


if __name__ == "__main__":
    main()