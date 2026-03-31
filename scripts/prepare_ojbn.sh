#!/usr/bin/env bash
# Mirror OJBench data from Hugging Face and build data/OJBN/test.jsonl.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"
bash "$PROJECT_ROOT/scripts/run_ojbn.sh" \
    --prepare-data \
    --download-all-data \
    "$@"
