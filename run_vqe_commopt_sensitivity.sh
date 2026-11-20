#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runner="$repo_root/run_commopt_parameter_analysis.sh"
base_config="$repo_root/config.commopt_parameter_analysis.json"
navi_python="${NAVI_PYTHON:-python}"
seeds_spec="42,43,44"
experiment_prefix="vqe_nofs"

usage() {
  cat <<'EOF'
Usage:
  ./run_vqe_commopt_sensitivity.sh [options]

Options:
  --seeds LIST              Comma-separated network seeds (default: 42,43,44)
  --experiment-prefix NAME  Experiment prefix (default: vqe_nofs)
  -h, --help                Show this help

The script runs VQE on the complete W={1,2,4}, L_max={5,10,15} grid with
greedy-table SliceOpt, max_match, the full-span safeguard disabled, the
TeleGate cache disabled, and reusable zero-cut/max-match prefixes enabled.
Experiments run only when this script is invoked explicitly.

Examples:
  ./run_vqe_commopt_sensitivity.sh --seeds 42 --experiment-prefix vqe_nofs
  ./run_vqe_commopt_sensitivity.sh --seeds 43,44 --experiment-prefix vqe_nofs
EOF
}

require_value() {
  local option="$1"
  local value="${2:-}"
  if [[ -z "$value" ]]; then
    echo "[ERROR] $option requires a value." >&2
    usage >&2
    exit 2
  fi
}

while (( $# > 0 )); do
  case "$1" in
    --seeds)
      require_value "$1" "${2:-}"
      seeds_spec="$2"
      shift 2
      ;;
    --experiment-prefix)
      require_value "$1" "${2:-}"
      experiment_prefix="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERROR] Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ ! "$experiment_prefix" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "[ERROR] --experiment-prefix may contain only letters, digits, dot, underscore, and hyphen." >&2
  exit 2
fi

IFS=',' read -r -a seeds <<< "$seeds_spec"
if (( ${#seeds[@]} == 0 )); then
  echo "[ERROR] --seeds must contain at least one network seed." >&2
  exit 2
fi

declare -A seen_seeds=()
experiment_ids=()
for seed in "${seeds[@]}"; do
  if [[ ! "$seed" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "[ERROR] --seeds entries must be nonnegative integers: ${seed:-<empty>}" >&2
    exit 2
  fi
  if [[ -n "${seen_seeds[$seed]:-}" ]]; then
    echo "[ERROR] --seeds contains a duplicate seed: $seed" >&2
    exit 2
  fi
  seen_seeds["$seed"]=1
  experiment_id="${experiment_prefix}_netseed${seed}"
  experiment_dir="$repo_root/outputs/navi_param_runs/$experiment_id"
  if [[ -e "$experiment_dir" ]]; then
    echo "[ERROR] Experiment output already exists; nothing was overwritten: $experiment_dir" >&2
    echo "[INFO] Choose another --experiment-prefix or request only seeds that have not run yet." >&2
    exit 2
  fi
  experiment_ids+=("$experiment_id")
done

if [[ ! -x "$runner" ]]; then
  echo "[ERROR] Parameter runner is not executable: $runner" >&2
  exit 2
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "[ERROR] jq is required to validate the fixed VQE configuration." >&2
  exit 2
fi
if [[ ! -f "$base_config" ]]; then
  echo "[ERROR] Base configuration not found: $base_config" >&2
  exit 2
fi
if ! jq -e \
  '(.compiler_config.navi.partition_assigner == "max_match")
   and (.compiler_config.navi.slice_strategy == "greedy_table")
   and ((.compiler_config.navi.seed // 26) == 26)' \
  "$base_config" >/dev/null; then
  echo "[ERROR] VQE sensitivity requires greedy_table, partition_assigner=max_match, and Telegate seed 26." >&2
  exit 2
fi
if ! "$navi_python" -c 'import sys; assert sys.version_info >= (3, 10)' >/dev/null 2>&1; then
  echo "[ERROR] NAVI_PYTHON must point to Python 3.10+." >&2
  exit 2
fi

for index in "${!seeds[@]}"; do
  seed="${seeds[$index]}"
  experiment_id="${experiment_ids[$index]}"
  echo "[VQE_SENSITIVITY] Running network seed $seed as $experiment_id"
  NAVI_PYTHON="$navi_python" "$runner" \
    --experiment-id "$experiment_id" \
    --qubits 80 \
    --cores 4 \
    --capacity 20 \
    --network all_to_all \
    --network-seed "$seed" \
    --benchmarks VQE \
    --widths 1,2,4 \
    --spans 5,10,15 \
    --telegate-cache off \
    --full-span-safeguard off \
    --reuse-shared-prefix on
done

summary_dir="$repo_root/outputs/vqe_commopt_sensitivity/$experiment_prefix"
"$navi_python" "$repo_root/scripts/summarize_vqe_commopt_sensitivity.py" \
  --input-csv "$repo_root/outputs/navi_param.csv" \
  --experiment-prefix "$experiment_prefix" \
  --output-dir "$summary_dir"

echo "[VQE_SENSITIVITY] Summary: $summary_dir"
