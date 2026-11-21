#!/bin/bash
set -euo pipefail
mkdir -p "$(dirname /app/out/answer.txt)"
cat > /app/out/answer.txt <<'ORACLE'
ANSWER: C) Sentiment (SST-2): is a single movie-review sentence positive or negative
ORACLE
