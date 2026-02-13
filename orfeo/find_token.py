from transformers import AutoTokenizer
import os

model_name=  "/orfeo/cephfs/scratch/dssc/gioluc/models/Llama-3.1-8B"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

token_id = 24149
token_str = tokenizer.convert_ids_to_tokens(token_id)

print(token_str)