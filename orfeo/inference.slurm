#!/bin/bash
#SBATCH -p GPU
#SBATCH -A dssc
#SBATCH --gres=gpu:V100:1
#SBATCH --time=00:30:00
#SBATCH --job-name=infer_llama
#SBATCH --output=./log/infer_llama-%j.out
#SBATCH --error=./log/infer_llama-%j.err

set -euo pipefail

echo "=== Job info ==="
echo "Date:        $(date)"
echo "Host:        $(hostname)"
echo "JobID:       ${SLURM_JOB_ID:-}"
echo "Partition:   ${SLURM_JOB_PARTITION:-}"
echo "Nodelist:    ${SLURM_NODELIST:-}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "================"

# activate venv on the compute node
source /orfeo/cephfs/home/ipauser/gioluc/venvs/llama/bin/activate

# model path
export MODEL_DIR="/orfeo/cephfs/scratch/dssc/gioluc/models/Llama-3.1-8B"

# prompt
export PROMPT="apple bird car egg house milk plane opera box sand sun mango rock math code phone"

# Esegui lo script
python /orfeo/cephfs/home/ipauser/gioluc/nlp/infer_llama.py