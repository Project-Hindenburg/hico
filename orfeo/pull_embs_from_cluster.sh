#!/usr/bin/env bash
set -euo pipefail

REMOTE_USER="gioluc"
REMOTE_HOST="orfeo"
REMOTE_DIR="/orfeo/cephfs/scratch/dssc/vdestasio/embeddings/"
LOCAL_DIR="./embeddings/new/"

rsync -avz \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR%/}/" \
  "${LOCAL_DIR%/}/"