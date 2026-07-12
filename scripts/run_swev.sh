#!/usr/bin/env bash
# Run the SWE-bench Verified benchmark (Phase 1: Agent Inference).
# This produces a predictions JSONL file. Evaluate it with Docker:
#   bash scripts/run_swev_eval.sh <predictions_file>
#
# Usage:
#   bash scripts/run_swev.sh                            # Full run (500 instances)
#   bash scripts/run_swev.sh --limit 5                  # First 5 instances
#   bash scripts/run_swev.sh --dry-run                  # Dry run
#   bash scripts/run_swev.sh --workers 4                # Parallel inference
#   bash scripts/run_swev.sh --filter 'django__.*'      # Regex filter by instance_id
#   bash scripts/run_swev.sh --slice 0:50               # Slice after filtering/shuffle
#   bash scripts/run_swev.sh --repo-cache-dir /tmp/repos  # Cache cloned repos
#   bash scripts/run_swev.sh --resume data/_results/prev.jsonl  # Resume from crash
#   bash scripts/run_swev.sh --preds-path data/_results/preds.json  # Skip IDs already predicted
#   bash scripts/run_swev.sh --task-timeout 1800           # 30min per task

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load environment variables
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

echo "=== SWE-bench Verified Benchmark (Phase 1: Agent Inference) ==="
echo "Project root: $PROJECT_ROOT"
echo ""

cd "$PROJECT_ROOT"
python -c "
import sys, types
from pathlib import Path
CODE_DIR = Path('code')
pkg = types.ModuleType('hello_agents')
pkg.__path__ = [str(CODE_DIR)]
pkg.__file__ = str(CODE_DIR / '__init__.py')
sys.modules['hello_agents'] = pkg
from hello_agents.benchmark.swev_bench import main
sys.argv = ['swev_bench'] + sys.argv[1:]
main()
" \
    --data-path "$PROJECT_ROOT/data/SWEV/test.jsonl" \
    --output-dir "$PROJECT_ROOT/data/_results" \
    --repo-cache-dir "$PROJECT_ROOT/data/_repo_cache" \
    --workers "${SWEV_WORKERS:-1}" \
    --max-steps 128 \
    "$@"
