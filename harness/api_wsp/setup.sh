#!/bin/bash

set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

# collect_outputs.py is stdlib only; the reply's python block runs with the task image's python3 and numpy.
if ! command -v python3 &> /dev/null; then
  apt-get update && apt-get install -y python3 || yum install -y python3 || apk add --no-cache python3
fi

mkdir -p /logs
