import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = os.environ["MODEL_DIR"]
TEXT_PATH = Path(os.environ["TEXT_PATH"])
LAST_N = int(os.environ.get("LAST_N", "50"))
LAYER_NUM = int(os.environ.get("LAYER_NUM", "26"))
out_dir = Path(os.environ.get("OUT_DIR", "."))

input_file_name = TEXT_PATH.stem

# Leggi righe = batch
lines = [ln.rstrip("\n") for ln in TEXT_PATH.read_text(encoding="utf-8").splitlines()]
# opzionale: rimuovi righe vuote
lines = [ln for ln in lines if ln.strip() != ""]
if len(lines) == 0:
    raise ValueError("Il file non contiene righe non-vuote.")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)

# Se il tokenizer non ha pad_token, lo impostiamo (comune con LLaMA)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    dtype=torch.float16,
    device_map="auto",
)
model.eval()

# 2) Tokenizza batch con padding (NO truncation)
# return_attention_mask=True è importante per sapere la lunghezza reale di ogni riga
inputs = tokenizer(
    lines,
    return_tensors="pt",
    padding=True,            # pad a max_len nel batch
    truncation=False,
    return_attention_mask=True,
)

device = next(model.parameters()).device
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs, output_hidden_states=True, return_dict=True)

hidden_states = outputs.hidden_states
# shape: [batch, seq_len, hidden]
layer_reps = hidden_states[LAYER_NUM]

input_ids_all = inputs["input_ids"]
attn_mask_all = inputs.get("attention_mask", None)

out_dir.mkdir(parents=True, exist_ok=True)

# 3) Per ogni elemento del batch (cioè ogni riga), estrai ultimi LAST_N token reali e salva
for b in range(layer_reps.shape[0]):
    if attn_mask_all is None:
        # fallback: se non c'è attention_mask, usiamo tutta la seq_len
        real_len = layer_reps.shape[1]
    else:
        # numero di token "veri" (non pad)
        real_len = int(attn_mask_all[b].sum().item())

    start = max(0, real_len - LAST_N)
    end = real_len

    reps_last = layer_reps[b, start:end].detach().to("cpu").to(torch.float16)   # [<=50, hidden]
    input_ids_last = input_ids_all[b, start:end].detach().to("cpu")            # [<=50]

    save_obj = {
        "input_ids_last": input_ids_last,
        "embeddings_last": reps_last,
        "batch_index": b
    }

    out_path = out_dir / f"reprs_{input_file_name}_b{b:04d}_layer{LAYER_NUM}.pt"
    torch.save(save_obj, out_path)

print(f"Salvati {layer_reps.shape[0]} file in: {out_dir}")
print(f"Layer {LAYER_NUM} | LAST_N={LAST_N}")