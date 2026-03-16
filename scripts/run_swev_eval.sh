#!/usr/bin/env bash
# Evaluate SWE-bench predictions using the official Docker harness.
#
# Usage:
#   bash scripts/run_swev_eval.sh <predictions_file>
#   bash scripts/run_swev_eval.sh <predictions_file> --max_workers 8
#   bash scripts/run_swev_eval.sh <predictions_file> --instance_ids django__django-16139

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SWEBENCH_ROOT="/home/kemove/LLM_Projects/SWE-bench"

# --- Check prerequisites ---

if [ $# -lt 1 ]; then
    echo "Usage: bash scripts/run_swev_eval.sh <predictions_file> [extra args...]"
    echo ""
    echo "Example:"
    echo "  bash scripts/run_swev_eval.sh data/_results/swev_predictions_20260316_120000.jsonl"
    echo "  bash scripts/run_swev_eval.sh data/_results/swev_predictions_*.jsonl --max_workers 8"
    exit 1
fi

PREDICTIONS_FILE="$1"
shift  # remaining args passed to run_evaluation

if [ ! -f "$PREDICTIONS_FILE" ]; then
    echo "ERROR: Predictions file not found: $PREDICTIONS_FILE"
    exit 1
fi

if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker is not running. Start Docker first."
    exit 1
fi

if [ ! -d "$SWEBENCH_ROOT/swebench" ]; then
    echo "ERROR: SWE-bench repo not found at $SWEBENCH_ROOT"
    echo "Clone it: git clone https://github.com/SWE-bench/SWE-bench.git $SWEBENCH_ROOT"
    exit 1
fi

# --- Run evaluation ---

RUN_ID="whale-code-$(date +%Y%m%d_%H%M%S)"

echo "=== SWE-bench Docker Evaluation ==="
echo "  Predictions: $PREDICTIONS_FILE"
echo "  Run ID:      $RUN_ID"
echo "  SWE-bench:   $SWEBENCH_ROOT"
echo ""

export PYTHONPATH="$SWEBENCH_ROOT:${PYTHONPATH:-}"

python -m swebench.harness.run_evaluation \
    --dataset_name princeton-nlp/SWE-bench_Verified \
    --split test \
    --predictions_path "$PREDICTIONS_FILE" \
    --run_id "$RUN_ID" \
    --max_workers 4 \
    --cache_level env \
    --clean True \
    --timeout 1800 \
    "$@"

echo ""
echo "=== Evaluation complete ==="
echo "  Run ID: $RUN_ID"
echo "  Logs:   logs/run_evaluation/$RUN_ID/"
