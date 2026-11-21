#!/bin/bash
set -euo pipefail
mkdir -p "$(dirname /app/out/answer.txt)"
cat > /app/out/answer.txt <<'ORACLE'
ANSWER: A) QNLI: does a sentence contain the answer to a question
ORACLE
