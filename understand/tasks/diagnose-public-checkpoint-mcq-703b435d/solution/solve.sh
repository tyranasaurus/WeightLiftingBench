#!/bin/bash
set -euo pipefail
mkdir -p "$(dirname /app/out/answer.txt)"
cat > /app/out/answer.txt <<'ORACLE'
ANSWER: D) STS-B: how similar in meaning are two sentences, scored 0 to 5
ORACLE
