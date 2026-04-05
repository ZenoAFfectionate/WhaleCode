#!/usr/bin/env bash
# Run the LiveCodeBench v6 benchmark.
# Usage:
#   bash scripts/run_lcb6.sh                    # Full run
#   bash scripts/run_lcb6.sh --limit 10         # First 10 tasks
#   bash scripts/run_lcb6.sh --dry-run          # Dry run
#   bash scripts/run_lcb6.sh --model gpt-4o     # Override model
#   bash scripts/run_lcb6.sh --resume data/_results/lcb6_xxx.jsonl
#                                             # Resume into a specific results file; rerun results replace same-task records

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Keep the benchmark inside the active Python environment only.
# This prevents ~/.local site-packages from shadowing conda packages such as rich.
export PYTHONNOUSERSITE=1

echo "=== LiveCodeBench v6 Benchmark ==="
echo "Project root: $PROJECT_ROOT"
echo "Python user site: disabled (PYTHONNOUSERSITE=1)"
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
from hello_agents.benchmark.lcb6_bench import main
sys.argv = ['lcb6_bench'] + sys.argv[1:]
main()
" \
    --data-path "$PROJECT_ROOT/data/LCB6/test.jsonl" \
    --output-dir "$PROJECT_ROOT/data/_results" \
    "$@"
