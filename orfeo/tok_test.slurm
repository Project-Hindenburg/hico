#!/bin/bash
#SBATCH -p GPU
#SBATCH -A dssc
#SBATCH --gres=gpu:V100:1
#SBATCH --time=00:30:00
#SBATCH --job-name=tok_test
#SBATCH --output=./log/tok-test-%j.out
#SBATCH --error=./log/tok-test-%j.err

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
export MODEL_DIR=/orfeo/cephfs/scratch/dssc/gioluc/models/Llama-3.1-8B

# text path
# export TEXT_PATH=/orfeo/cephfs/scratch/dssc/vdestasio/input_context/one_random_walk/grid_16/grid_dataset_one_rw_300.txt

# Esegui lo script
python /orfeo/cephfs/home/ipauser/gioluc/nlp/scripts/tokenizer_test.py