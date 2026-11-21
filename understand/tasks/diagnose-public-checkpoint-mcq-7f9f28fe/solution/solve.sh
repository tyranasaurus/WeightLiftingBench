#!/bin/bash
set -euo pipefail
mkdir -p "$(dirname /app/out/answer.txt)"
cat > /app/out/answer.txt <<'ORACLE'
ANSWER: A) Udnie by Francis Picabia, an early abstract cubist painting
ORACLE
