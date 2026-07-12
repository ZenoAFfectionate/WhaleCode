#!/usr/bin/env bash
# Run the Terminal Bench 2.0 benchmark.
# Usage:
#   bash scripts/run_term.sh                    # Full run
#   bash scripts/run_term.sh --limit 5          # First 5 tasks
#   bash scripts/run_term.sh --dry-run          # Dry run
#   bash scripts/run_term.sh --resume data/_results/term_bench_2_xxx.jsonl

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# Keep benchmark deps inside the active Python environment only.
export PYTHONNOUSERSITE=1

echo "=== Terminal Bench 2.0 Benchmark ==="
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
from hello_agents.benchmark.term_bench import main
sys.argv = ['term_bench'] + sys.argv[1:]
main()
" \
    --data-path "$PROJECT_ROOT/data/TERM/test.jsonl" \
    --output-dir "$PROJECT_ROOT/data/_results" \
    --trajectory-dir "$PROJECT_ROOT/data/_trajectory" \
    "$@"
