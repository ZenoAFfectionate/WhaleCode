#!/usr/bin/env bash
# Run the AIME benchmark.
# Usage:
#   bash scripts/run_aime.sh                    # Full run
#   bash scripts/run_aime.sh --limit 10         # First 10 tasks
#   bash scripts/run_aime.sh --dry-run          # Dry run
#   bash scripts/run_aime.sh --model gpt-4o     # Override model

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load environment variables
if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

echo "=== AIME Benchmark ==="
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
from hello_agents.benchmark.aime_bench import main
sys.argv = ['aime_bench'] + sys.argv[1:]
main()
" \
    --data-path "$PROJECT_ROOT/data/AIME/test.jsonl" \
    --output-dir "$PROJECT_ROOT/data/_results" \
    --timeout 60 \
    "$@"
