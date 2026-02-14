#!/bin/bash
set -euo pipefail

BASE_IN="/orfeo/cephfs/scratch/dssc/vdestasio/input_context"
OUT_LIST="${1:-filelist.txt}"

# Prende solo i *_300.txt, *_600.txt, *_1200.txt, *_1800.txt
find "$BASE_IN" -type f \( -name "*_300.txt" -o -name "*_600.txt" -o -name "*_1200.txt" -o -name "*_1800.txt" \) \
  | sort > "$OUT_LIST"

echo "Scritto $(wc -l < "$OUT_LIST") file in $OUT_LIST"


# how to use
# bash make_filelist.sh /orfeo/cephfs/scratch/dssc/vdestasio/input_context_filelist.txt

