#!/bin/bash
set -euo pipefail
mkdir -p "$(dirname /app/out/answer.txt)"
cat > /app/out/answer.txt <<'ORACLE'
ANSWER: C) Job-posting task statements: does a line from a job posting describe a task performed in the job
ORACLE
