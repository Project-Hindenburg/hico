import argparse
import os
from pathlib import Path
from typing import Iterable, List, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--input-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)

    # Se vuoi: usa una filelist invece di scandire la directory
    ap.add_argument("--filelist", type=Path, default=None)

    # Default come hai descritto tu
    ap.add_argument("--patterns", nargs="+",
                    default=["*_3000.txt"])

                    # default=["*_300.txt", "*_600.txt", "*_1200.txt", "*_1800.txt"])

    # ap.add_argument("--last-n", type=int, default=500)
    ap.add_argument("--layer-num", type=int, default=26)

    # Batching interno (per righe dentro un file)
    ap.add_argument("--batch-size", type=int, default=8)

    # Se vuoi ripartire senza rifare tutto
    ap.add_argument("--skip-if-exists", action="store_true")

    args = ap.parse_args()

    model_dir: Path = args.model_dir
    input_root: Path = args.input_root
    output_root: Path = args.output_root
    # last_n: int = args.last_n
    layer_num: int = args.layer_num
    batch_size: int = args.batch_size

    if args.filelist is not None:
        files = [Path(x.strip()) for x in args.filelist.read_text(encoding="utf-8").splitlines() if x.strip()]
    else:
        files = sorted(set(iter_input_files(input_root, args.patterns)))

    if not files:
        raise ValueError("Nessun file trovato (controlla input_root/patterns o filelist).")

    print(f"Trovati {len(files)} file input.")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model
    model = AutoModel.from_pretrained(
        str(model_dir),
        dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    device = next(model.parameters()).device
    print(f"Device principale: {device}")

    # Inference mode
    with torch.inference_mode():
        for fpath in files:
            try:
                rel_dir = fpath.parent.relative_to(input_root)  # es: one_random_walk/expA/...
            except ValueError:
                # Se per qualche motivo il file non è sotto input_root
                rel_dir = Path()

            out_dir = output_root / rel_dir
            out_dir.mkdir(parents=True, exist_ok=True)

            lines = read_nonempty_lines(fpath)
            if not lines:
                print(f"[SKIP] {fpath} (only empty lines)")
                continue

            input_file_name = fpath.stem
            print(f"\n=== FILE: {fpath} | non empty lines: {len(lines)} | OUT: {out_dir} ===")

            global_line_offset = 0

            for chunk in batched(lines, batch_size=batch_size):
                # Tokenizza batch con padding, no truncation
                inputs = tokenizer(
                    chunk,
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                    return_attention_mask=True,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                hidden_states = outputs.hidden_states
                layer_reps = hidden_states[layer_num]  # [batch, seq_len, hidden]

                input_ids_all = inputs["input_ids"]
                attn_mask_all = inputs.get("attention_mask", None)

                # Per ogni riga del chunk salva ultimi LAST_N token reali
                for b in range(layer_reps.shape[0]):
                    if attn_mask_all is None:
                        real_len = layer_reps.shape[1]
                    else:
                        real_len = int(attn_mask_all[b].sum().item())

                    # start = max(0, real_len - last_n)
                    start = 0
                    end = real_len

                    reps_last = layer_reps[b, start:end].detach().to("cpu").to(torch.float16)
                    input_ids_last = input_ids_all[b, start:end].detach().to("cpu")

                    line_index = global_line_offset + b

                    save_obj = {
                        "input_ids_last": input_ids_last,
                        "embeddings_last": reps_last,
                        "line_index": line_index,
                        "source_file": str(fpath),
                        "layer_num": layer_num,
                        # "last_n": last_n,
                    }

                    out_path = out_dir / f"reprs_{input_file_name}_line{line_index:06d}_layer{layer_num}.pt"

                    if args.skip_if_exists and out_path.exists():
                        continue

                    torch.save(save_obj, out_path)

                global_line_offset += len(chunk)

            print(f"[OK] Embeddings saved for: {fpath}")

    print("\nDone.")


if __name__ == "__main__":
    main()
