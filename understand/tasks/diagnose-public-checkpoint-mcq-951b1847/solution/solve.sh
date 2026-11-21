#!/bin/bash
set -euo pipefail
mkdir -p "$(dirname /app/out/answer.txt)"
cat > /app/out/answer.txt <<'ORACLE'
ANSWER: A) Sentiment (IMDB): is a full-length IMDb movie review positive or negative
ORACLE
