#!/bin/bash
set -euo pipefail
mkdir -p "$(dirname /app/out/answer.txt)"
cat > /app/out/answer.txt <<'ORACLE'
ANSWER: B) Pointilism, a pointillist painting built from small separate dots of colour
ORACLE
