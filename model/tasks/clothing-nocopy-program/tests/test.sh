#!/bin/bash
set -euo pipefail
python /tests/score_program.py --config /tests/verifier_config.json --reward-path /logs/verifier/reward.json
