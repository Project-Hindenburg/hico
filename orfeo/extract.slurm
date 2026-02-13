#!/bin/bash
#SBATCH -p GPU
#SBATCH -A dssc
#SBATCH --gres=gpu:V100:1
#SBATCH --time=00:30:00
#SBATCH --job-name=infer_llama
#SBATCH --output=./log/extract_repr-%j.out
#SBATCH --error=./log/extract_repr-%j.err

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

export MODEL_DIR=/orfeo/cephfs/scratch/dssc/gioluc/models/Llama-3.1-8B
export TEXT_PATH=/orfeo/cephfs/scratch/dssc/vdestasio/input_context/paper_grid_one_rw_1400.txt
export OUT_DIR=/orfeo/cephfs/scratch/dssc/vdestasio/embeddings

export LAST_N=500
export LAYER_NUM=26

python /orfeo/cephfs/home/ipauser/gioluc/nlp/extract_embeddings.py
