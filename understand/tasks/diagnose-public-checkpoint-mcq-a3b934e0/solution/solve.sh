#!/bin/bash
set -euo pipefail
mkdir -p "$(dirname /app/out/answer.txt)"
cat > /app/out/answer.txt <<'ORACLE'
ANSWER: B) Prompt injection: is a prompt sent to an LLM an injection attack
ORACLE
