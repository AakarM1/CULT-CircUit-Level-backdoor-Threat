#!/usr/bin/env bash
# Master runner. Sequential by default; pass --parallel to background each.
# Skip flags: --skip-exp1 ... --skip-exp6.
# Forwards extras (e.g. --dry-run, --epochs N) to each experiment script.
set -uo pipefail

PARALLEL=0
SKIP=()
PASS=()

for arg in "$@"; do
  case "$arg" in
    --parallel) PARALLEL=1 ;;
    --skip-exp1|--skip-exp2|--skip-exp3|--skip-exp4|--skip-exp5|--skip-exp6)
      SKIP+=("${arg#--skip-}") ;;
    *) PASS+=("$arg") ;;
  esac
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT/experiments" || exit 1

# Auto-detect python
if [[ -z "${PYTHON:-}" ]]; then
  if command -v python >/dev/null 2>&1; then PYTHON=python
  elif command -v python3 >/dev/null 2>&1; then PYTHON=python3
  else echo "ERROR: no python on PATH" >&2; exit 1; fi
fi

is_skipped() {
  local exp="$1"
  for s in "${SKIP[@]:-}"; do [[ "$s" == "$exp" ]] && return 0; done
  return 1
}

run_exp() {
  local exp="$1"
  local script="run_${exp}.py"
  if is_skipped "$exp"; then
    echo "[SKIP] $exp"
    return
  fi
  echo "============================================================"
  echo "[START] $exp  (parallel=$PARALLEL)"
  echo "============================================================"
  if [[ "$PARALLEL" -eq 1 ]]; then
    "$PYTHON" "$script" "${PASS[@]:-}" &
  else
    "$PYTHON" "$script" "${PASS[@]:-}"
  fi
}

for exp in exp1 exp2 exp3 exp4 exp5 exp6; do
  run_exp "$exp"
done

if [[ "$PARALLEL" -eq 1 ]]; then
  wait
fi

echo "[DONE] All experiments completed."
