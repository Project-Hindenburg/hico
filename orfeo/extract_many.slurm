#!/bin/bash
#SBATCH -p GPU
#SBATCH -A dssc
#SBATCH --gres=gpu:V100:1
#SBATCH --time=01:00:00
#SBATCH --job-name=infer_llama_many
#SBATCH --output=/u/ipauser/gioluc/log/extract_many-%j.out
#SBATCH --error=/u/ipauser/gioluc/log/extract_many-%j.err

set -euo pipefail

source /orfeo/cephfs/home/ipauser/gioluc/venvs/llama/bin/activate

MODEL_DIR=/orfeo/cephfs/scratch/dssc/gioluc/models/Llama-3.1-8B
INPUT_ROOT=/orfeo/cephfs/scratch/dssc/vdestasio/input_context/
# INPUT_ROOT=/orfeo/cephfs/scratch/dssc/vdestasio/input_context/one_random_walk/grid_16/
OUTPUT_ROOT=/orfeo/cephfs/scratch/dssc/vdestasio/embeddings

python /orfeo/cephfs/home/ipauser/gioluc/nlp/scripts/extract_embeddings_many.py \
  --model-dir "$MODEL_DIR" \
  --input-root "$INPUT_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --last-n 500 \
  --layer-num 26 \
  --batch-size 1 \
  --skip-if-exists
