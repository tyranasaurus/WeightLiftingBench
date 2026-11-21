#!/bin/bash
set -euo pipefail
mkdir -p "$(dirname /app/out/answer.txt)"
cat > /app/out/answer.txt <<'ORACLE'
ANSWER: D) Rain Princess by Leonid Afremov, a palette-knife painting of a rainy evening street
ORACLE
