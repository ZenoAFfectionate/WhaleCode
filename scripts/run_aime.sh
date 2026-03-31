set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ -f "$PROJECT_ROOT/.env" ]; then
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

year=""
has_data_path=0
has_resume=0
declare -a passthrough_args=()

while [ $# -gt 0 ]; do
    case "$1" in
        --year)
            if [ $# -lt 2 ]; then
                echo "Missing value for --year" >&2
                exit 1
            fi
            year="$2"
            shift 2
            ;;
        --year=*)
            year="${1#*=}"
            shift
            ;;
        --data-path)
            if [ $# -lt 2 ]; then
                echo "Missing value for --data-path" >&2
                exit 1
            fi
            has_data_path=1
            passthrough_args+=("$1" "$2")
            shift 2
            ;;
        --data-path=*)
            has_data_path=1
            passthrough_args+=("$1")
            shift
            ;;
        --resume)
            if [ $# -lt 2 ]; then
                echo "Missing value for --resume" >&2
                exit 1
            fi
            has_resume=1
            passthrough_args+=("$1" "$2")
            shift 2
            ;;
        --resume=*)
            has_resume=1
            passthrough_args+=("$1")
            shift
            ;;
        *)
            passthrough_args+=("$1")
            shift
            ;;
    esac
done

run_aime() {
    local label="$1"
    shift

    echo "=== AIME Benchmark (${label}) ==="
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
        --output-dir "$PROJECT_ROOT/data/_results" \
        --timeout 60 \
        "$@" \
        "${passthrough_args[@]}"
}

if [ "$has_data_path" -eq 1 ]; then
    if [ -n "$year" ]; then
        run_aime "$year" --year "$year"
    else
        run_aime "custom"
    fi
    exit 0
fi

if [ -n "$year" ]; then
    run_aime "$year" --year "$year"
    exit 0
fi

if [ "$has_resume" -eq 1 ]; then
    echo "When using --resume, please also pass --year 24|25|26 or --data-path." >&2
    exit 1
fi

for run_year in 24 25 26; do
    run_aime "$run_year" --year "$run_year"
done
