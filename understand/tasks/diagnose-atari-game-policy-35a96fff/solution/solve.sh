#!/bin/bash
set -euo pipefail
mkdir -p "$(dirname /app/out/answer.txt)"
cat > /app/out/answer.txt <<'ORACLE'
ANSWER: Q*bert
ORACLE
