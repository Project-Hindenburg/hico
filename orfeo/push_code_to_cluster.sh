#!/usr/bin/env bash
set -euo pipefail

LOCAL_CODE_DIR="./orfeo"
REMOTE="gioluc@orfeo"
CODE_REMOTE_DIR="/u/ipauser/gioluc/nlp/scripts"

echo "Pushing code to cluster..."
rsync -avz --delete \
	--exclude '__pycache__/' \
	--exclude '*.pyc' \
	--exclude '.git/' \
	"${LOCAL_CODE_DIR%/}/" \
	"${REMOTE}:${CODE_REMOTE_DIR%/}/"