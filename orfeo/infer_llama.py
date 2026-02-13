import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name=  "/orfeo/cephfs/scratch/dssc/gioluc/models/Llama-3.1-8B"

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

tokenizer = AutoTokenizer.from_pretrained(model_name)

text = os.environ.get("PROMPT")

# tokenizer
inputs = tokenizer(text, return_tensors="pt")

input_ids = inputs["input_ids"]

print("Token IDs:", input_ids[0].tolist())

tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
print("Tokens:", tokens)

# embeddings

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     dtype=torch.float16,
#     device_map="auto"   
# )

# model.eval()

# inputs = {k: v.to(model.device) for k, v in inputs.items()}


# with torch.no_grad():
#     outputs = model(**inputs, output_hidden_states=True, return_dict=True)

# hidden_states = outputs.hidden_states

# print("\nNumero layer (incluso embedding):", len(hidden_states))
# print("Shape hidden state embedding:", hidden_states[0].shape)
# print("Shape ultimo layer:", hidden_states[-1].shape)

# # esempio: vettore ultimo token ultimo layer
# last_token_vec = hidden_states[-1][0, -1]
# print("Shape vettore ultimo token:", last_token_vec.shape)