import os
from pathlib import Path
from transformers import AutoTokenizer


MODEL_DIR = os.environ["MODEL_DIR"]
# TEXT_PATH = Path(os.environ["TEXT_PATH"])

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)

text = " January February March April May June July August September October November December"

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

token_ids = tokenizer(text, 
                      return_tensors="pt",
                      padding=True,
                      truncation=False,
                      return_attention_mask=True,
                      add_special_tokens=False
                      )["input_ids"][0]

# token_str = tokenizer.convert_ids_to_tokens(token_ids.tolist())

# print("Token IDs shape:\n", token_ids.shape)
# print("\nToken IDs:\n", list(token_ids))
# print("\nToken Strings shape:\n", len(token_str))
# print("\nToken Strings:\n", token_str)

for tid, token in zip(token_ids.tolist(), tokenizer.convert_ids_to_tokens(token_ids.tolist())):
    print(f"{token}\t{tid}")