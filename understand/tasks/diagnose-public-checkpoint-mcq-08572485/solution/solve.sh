#!/bin/bash
set -euo pipefail
mkdir -p "$(dirname /app/out/answer.txt)"
cat > /app/out/answer.txt <<'ORACLE'
ANSWER: D) writing restaurant descriptions from structured attribute lists
ORACLE
