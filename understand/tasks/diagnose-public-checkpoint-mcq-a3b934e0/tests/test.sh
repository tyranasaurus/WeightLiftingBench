#!/bin/bash
set -euo pipefail
python3 /tests/grade.py --answer /app/out/answer.txt --expected /tests/expected.json --reward /logs/verifier/reward.json
