#!/bin/bash
# Download the 14 data files of the Model sub-bench from Hugging Face and check them against SHA256SUMS.txt.
# Usage: bash fetch.sh [OUT_DIR]
set -euo pipefail
out=${1:-.}; here=$(cd "$(dirname "$0")" && pwd); mkdir -p "$out"
base=https://huggingface.co/datasets/tyranasaurus/WeightLiftingBench/resolve/main/model/data
auth=(); [ -n "${HF_TOKEN:-}" ] && auth=(-H "Authorization: Bearer $HF_TOKEN")
for task in digits clothing traffic galaxies letters cifar person; do
  for split in train test; do
    curl -fL --retry 3 "${auth[@]}" -o "$out/$task-$split.npz" "$base/$task-$split.npz"
  done
done
(cd "$out" && shasum -a 256 -c "$here/SHA256SUMS.txt")
