#!/usr/bin/env bash
# Run the OJBench benchmark on the Python split.
# Usage:
#   bash scripts/run_ojbn.sh --prepare-data --download-all-data
#   bash scripts/run_ojbn.sh --limit 5
#   bash scripts/run_ojbn.sh --judge-backend official --limit 1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

echo "=== OJBench Benchmark ==="
echo "Project root: $PROJECT_ROOT"
echo ""

cd "$PROJECT_ROOT"
python -c '
import sys, types
from pathlib import Path
CODE_DIR = Path("code")
pkg = types.ModuleType("hello_agents")
pkg.__path__ = [str(CODE_DIR)]
pkg.__file__ = str(CODE_DIR / "__init__.py")
sys.modules["hello_agents"] = pkg
from hello_agents.benchmark.ojbn_bench import main
sys.argv = ["ojbn_bench"] + sys.argv[1:]
main()
' \
    --data-path "$PROJECT_ROOT/data/OJBN/test.jsonl" \
    --raw-dir "$PROJECT_ROOT/data/OJBN/raw" \
    --output-dir "$PROJECT_ROOT/data/_results" \
    "$@"
