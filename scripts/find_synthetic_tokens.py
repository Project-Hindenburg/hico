from transformers import LlamaTokenizer
import re
from nltk.corpus import words, stopwords
import nltk
import random

# ─── Load the LLaMA‑3.2‑1B tokenizer ───
# TODO: download tokenizer and use it
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-hf")

# Get the vocabulary (token -> id)
vocab = tokenizer.get_vocab()
tokens = list(vocab.keys())

# ─── Filter tokens ───
# Keep only tokens that are "single" (not SentencePiece subwords)
single_tokens = [t for t in tokens if not t.startswith('▁') and not t.startswith('##')]

# Keep only alphabetic/simple tokens
simple_tokens = [t for t in single_tokens if re.fullmatch(r"[A-Za-z]+", t)]

# ─── Download word lists ───
nltk.download('words')
nltk.download('stopwords')

english_words = set(words.words())
common_words = set(stopwords.words('english'))

# Filter out common English words
rare_tokens = [
    t for t in simple_tokens
    if t.lower() not in english_words and t.lower() not in common_words
]

# ─── Random subset ───
subset_size = 50  # adjust as desired
sampled_tokens = random.sample(rare_tokens, subset_size)

print(f"Total filtered tokens: {len(rare_tokens)}")
print("Sampled tokens:", sampled_tokens[:50])