#!/bin/bash
set -euo pipefail
mkdir -p "$(dirname /app/out/answer.txt)"
cat > /app/out/answer.txt <<'ORACLE'
ANSWER: A) Question focus: is a question about a document generic (e.g. asks for a summary) or about specific content
ORACLE
