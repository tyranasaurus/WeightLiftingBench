#!/bin/bash
set -euo pipefail
mkdir -p /app/out
cp "$(dirname "$0")/build.py" /app/out/build.py
python3 /app/out/build.py /app/out
echo "Reference: data-free constant answer." > /app/out/method.md
