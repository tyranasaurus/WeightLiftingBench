#!/bin/bash
set -euo pipefail
mkdir -p "$(dirname /app/out/answer.txt)"
cat > /app/out/answer.txt <<'ORACLE'
ANSWER: A) MNLI: does a premise entail, contradict or say nothing about a hypothesis
ORACLE
