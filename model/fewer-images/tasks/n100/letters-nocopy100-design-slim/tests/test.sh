#!/bin/bash
set -euo pipefail
cd /tests
python /tests/score_design.py --reward-path /logs/verifier/reward.json
