#!/bin/bash
# Download each Understand task's weights file from Hugging Face into its task folder (tasks/<task>/environment/policy.npz)
# and check it against weights-SHA256SUMS.txt. Usage: bash fetch_weights.sh
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
base=https://huggingface.co/datasets/tyranasaurus/WeightLiftingBench/resolve/main/understand/weights
auth=(); [ -n "${HF_TOKEN:-}" ] && auth=(-H "Authorization: Bearer $HF_TOKEN")
while read -r sum file; do
  task=${file%.npz}; dest="$here/tasks/$task/environment/policy.npz"
  curl -fL --retry 3 "${auth[@]}" -o "$dest" "$base/$file"
  echo "$sum  $dest" | shasum -a 256 -c -
done < "$here/weights-SHA256SUMS.txt"
