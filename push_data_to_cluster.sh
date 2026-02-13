#!/usr/bin/env bash
set -euo pipefail

LOCAL_DATA_DIR="./data"
REMOTE="gioluc@orfeo"
DATA_REMOTE_DIR="/orfeo/cephfs/scratch/dssc/vdestasio/input_context"

echo "Pushing data to cluster..."
rsync -avz --delete \
	--no-times --omit-dir-times \
	--no-perms --no-owner --no-group \
  	--exclude '.git/' \
	--exclude 'uncorrelated-words/' \
	"${LOCAL_DATA_DIR%/}/" \
	"${REMOTE}:${DATA_REMOTE_DIR%/}/"