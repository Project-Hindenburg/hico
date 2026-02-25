# `scripts/uncorr_tkns.py` Documentation

## Purpose
`uncorr_tkns.py` selects a set of lowercase, single-token words that are as mutually dissimilar as possible in a chosen model layer.

It does this with four stages:
1. Filter raw words to valid lowercase alphabetic strings.
2. Keep only words that are single-token under the exact surface form `" " + word`.
3. Compute normalized hidden-state vectors for those token ids at a chosen layer.
4. Run greedy farthest-point sampling (max-min cosine diversity).


## CLI Arguments
Defined in `parse_args()` (`scripts/uncorr_tkns.py:25`):

- `--model-id`: model id/path argument (default `meta-llama/Llama-3.1-8B`).
- `--wordlist`: input word list path.
- `--layer`: hidden-state layer index (`0` is embedding output).
- `--k`: number of words to select.
- `--seed`: random seed for reproducibility.
- `--batch-size`: embedding forward-pass batch size.
- `--device`: device string for tensor placement.
- `--output`: output path for selected words.

Important implementation detail:
- In current code, the model is loaded from a hardcoded cluster path (`scripts/uncorr_tkns.py:150`), so `--model-id` is currently ignored.

## End-to-End Flow

### 1) Word loading and normalization
Implemented in `load_words()` (`scripts/uncorr_tkns.py:38`).

- Reads line-by-line from `--wordlist`.
- Lowercases each entry.
- Keeps only regex matches `^[a-z]+$`.

Consequence:
- Numbers, punctuation, mixed forms, and non-ASCII forms are excluded.

### 2) Single-token candidate construction
Implemented in `build_candidates()` (`scripts/uncorr_tkns.py:48`).

For each cleaned word `w`:
- Construct `surface = " " + w` (`scripts/uncorr_tkns.py:53`).
- Tokenize with `add_special_tokens=False`.
- Keep only if tokenized length is exactly 1 (`len(ids) == 1`).
- Deduplicate by token id (`seen_ids`), so each token id appears once.

Consequence:
- The script is strict about prompt surface form. A word that is single-token without leading space may still be rejected.

### 3) Layer embedding extraction
Implemented in `embed_at_layer()` (`scripts/uncorr_tkns.py:70`).

For each candidate token id:
- Build a short context sequence:
  - `[CLS, token, SEP]` if tokenizer provides `cls/sep`.
  - Else `[BOS, token]`.
  - Else `[EOS, token]`.
  - Else `[token]`.
- Forward pass with `output_hidden_states=True`.
- Extract vector at the selected token position and layer.
- L2-normalize all vectors (`normalize_rows()`).

Layer indexing:
- `layer >= 0`: direct index into `hidden_states`.
- `layer < 0`: Python-style from the end.
- Out-of-range layer raises a `ValueError` (`scripts/uncorr_tkns.py:109`).

Interpretation:
- After normalization, cosine similarity equals dot product.

### 4) Selection algorithm: greedy farthest-point sampling
Implemented in `farthest_point_sampling()` (`scripts/uncorr_tkns.py:124`).

Let normalized vectors be `v_i`.

Algorithm:
1. Choose one random seed index.
2. Maintain `max_sim[i] = max_{j in S} cos(v_i, v_j)` for each candidate `i`, where `S` is selected set.
3. Add the index with smallest `max_sim`.
4. Update `max_sim` with the new selected vector.
5. Repeat until `k` items are selected.

Key property:
- It minimizes each new point's similarity to its nearest selected neighbor (greedy max-min objective).
- This is not guaranteed to be globally optimal, but is a strong and standard approximation.

## Outputs

### A) Selected token file
Written to `--output` (`scripts/uncorr_tkns.py:179`).

Format per row:
- `<word>\t<token_id>`

### B) Cosine similarity matrix file
Written to `<output>.cosine.tsv` (`scripts/uncorr_tkns.py:184`).

Format:
- Header: `word` + selected word labels.
- Each row: row word + cosine similarities against all selected words.

Expected checks:
- Diagonal should be ~`1.0`.
- Off-diagonal should be as low as possible for your target diversity.

## What “maximally dissimilar” means here
In this script, it means:
- **Greedy max-min under cosine similarity** in the chosen layer representation space.
- Not semantic dissimilarity from human judgment directly.
- Not exact global combinatorial optimum.

## Complexity and Resource Profile
Let:
- `N` = number of candidates after single-token filtering,
- `D` = hidden size,
- `K` = selected set size.

Main costs:
- Embedding extraction: roughly `O(N * model_forward)`.
- Selection loop similarity updates: roughly `O(K * N * D)`.
- Memory for vectors: `O(N * D)` float32 in current implementation.

Practical implication:
- Large `N` with large `D` can cause OOM/kill on constrained nodes.

## Known Limitations in Current File
Observed in current code:
- `--model-id` is not used at load time because of hardcoded `model_name` (`scripts/uncorr_tkns.py:150`).
- Model class is `AutoModelForCausalLM` (`scripts/uncorr_tkns.py:20`), so encoder models (e.g., BERT) are not handled by the current version.
- Output row currently omits the exact surface string and stores only `word` + `token_id` (`scripts/uncorr_tkns.py:182`).
- Full candidate embedding matrix is materialized in memory.

## Reproducibility Notes
The script sets seeds in `main()` (`scripts/uncorr_tkns.py:146`):
- `random.seed`
- `numpy.random.seed`
- `torch.manual_seed`

This makes the random starting point deterministic for the same environment and model/tokenizer versions.

## Example Command (as implemented now)
```bash
python /Users/giaco/Documents/projects/NLP/hico/scripts/uncorr_tkns.py \
  --wordlist /Users/giaco/Documents/projects/NLP/hico/data/english.txt \
  --layer 26 \
  --k 64 \
  --device cuda \
  --output /Users/giaco/Documents/projects/NLP/hico/outputdir/selected_llama31_layer26.txt
```

## Quick Validation Checklist
1. Confirm candidate count printed by script is large enough for your `k`.
2. Open `<output>.cosine.tsv`.
3. Verify diagonal entries are `1.000000`.
4. Inspect largest off-diagonal entries; these are your most similar selected pairs.
5. If off-diagonal values are too high, reduce `k` or change `layer`.
