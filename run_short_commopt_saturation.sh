#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runner="$repo_root/run_commopt_parameter_analysis.sh"
base_config="$repo_root/config.commopt_parameter_analysis.json"
validation_report="$repo_root/outputs/navi_param.validation.json"
navi_python="${NAVI_PYTHON:-python}"
seeds_spec="42,43,44"
experiment_prefix="commopt_short_nofs"
dry_run=false

benchmarks=(SwapTest Comparator CuccaroAdder QV Random)

usage() {
  cat <<'EOF'
Usage:
  ./run_short_commopt_saturation.sh [options]

Options:
  --seeds LIST              Comma-separated network seeds (default: 42,43,44)
  --experiment-prefix NAME  Experiment prefix (default: commopt_short_nofs)
  --dry-run                 Print the ordered jobs without running experiments
  -h, --help                Show this help

The targeted W=1 grid is:
  SwapTest, Comparator, CuccaroAdder, QV: L_max={5,10}
  Random:                                  L_max={5,10,15}

Runs are benchmark-major and sequential, with network seeds 42, 43, and 44 by
default. Every job uses 80 qubits, 4x20 all-to-all QPUs, greedy-table SliceOpt,
max_match, disabled full-span safeguard and TeleGate cache, and shared-prefix
reuse. L=15 is omitted when K<=10 because it adds no candidates beyond L=10.
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
    --dry-run)
      dry_run=true
      shift
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
done

if [[ ! -x "$runner" ]]; then
  echo "[ERROR] Parameter runner is not executable: $runner" >&2
  exit 2
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "[ERROR] jq is required to validate the fixed saturation configuration." >&2
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
  echo "[ERROR] Saturation runs require greedy_table, partition_assigner=max_match, and Telegate seed 26." >&2
  exit 2
fi
if ! "$navi_python" -c 'import sys; assert sys.version_info >= (3, 10)' >/dev/null 2>&1; then
  echo "[ERROR] NAVI_PYTHON must point to Python 3.10+." >&2
  exit 2
fi

spans_for() {
  if [[ "$1" == "Random" ]]; then
    printf '5,10,15'
  else
    printf '5,10'
  fi
}

experiment_id_for() {
  local benchmark="$1"
  local seed="$2"
  printf '%s_%s_netseed%s' "$experiment_prefix" "${benchmark,,}" "$seed"
}

experiment_is_complete() {
  local experiment_id="$1"
  local benchmark="$2"
  [[ -f "$validation_report" ]] || return 1
  jq -e \
    --arg experiment_id "$experiment_id" \
    --arg benchmark "$benchmark" \
    '([.invalid_runs[]? | select(startswith($experiment_id + "/"))] | length) == 0
     and ([.groups[]?
           | select(.experiment_id == $experiment_id and .benchmark == $benchmark)
           | select(.complete == true and (.errors | length == 0))]
          | length) == 1' \
    "$validation_report" >/dev/null
}

declare -A skip_jobs=()
for benchmark in "${benchmarks[@]}"; do
  for seed in "${seeds[@]}"; do
    experiment_id="$(experiment_id_for "$benchmark" "$seed")"
    experiment_dir="$repo_root/outputs/navi_param_runs/$experiment_id"
    if [[ -e "$experiment_dir" ]]; then
      if experiment_is_complete "$experiment_id" "$benchmark"; then
        skip_jobs["$experiment_id"]=1
      else
        echo "[ERROR] Existing experiment is incomplete or invalid; nothing was overwritten:" >&2
        echo "        $experiment_dir" >&2
        echo "[INFO] Inspect it or choose a new --experiment-prefix." >&2
        exit 2
      fi
    fi
  done
done

job_number=0
job_count=$(( ${#benchmarks[@]} * ${#seeds[@]} ))
for benchmark in "${benchmarks[@]}"; do
  spans="$(spans_for "$benchmark")"
  echo "[SHORT_SATURATION] Benchmark: $benchmark, W=1, L_max={$spans}"
  for seed in "${seeds[@]}"; do
    job_number=$((job_number + 1))
    experiment_id="$(experiment_id_for "$benchmark" "$seed")"
    if [[ -n "${skip_jobs[$experiment_id]:-}" ]]; then
      echo "[SHORT_SATURATION] [$job_number/$job_count] SKIP complete $experiment_id"
      continue
    fi
    if [[ "$dry_run" == true ]]; then
      echo "[SHORT_SATURATION] [$job_number/$job_count] DRY-RUN $benchmark network_seed=$seed W=1 L_max={$spans} experiment_id=$experiment_id"
      continue
    fi

    echo "[SHORT_SATURATION] [$job_number/$job_count] START $benchmark network_seed=$seed experiment_id=$experiment_id"
    NAVI_PYTHON="$navi_python" "$runner" \
      --experiment-id "$experiment_id" \
      --qubits 80 \
      --cores 4 \
      --capacity 20 \
      --network all_to_all \
      --network-seed "$seed" \
      --benchmarks "$benchmark" \
      --widths 1 \
      --spans "$spans" \
      --min-atomic-subcircuits 0 \
      --telegate-cache off \
      --full-span-safeguard off \
      --reuse-shared-prefix on
    echo "[SHORT_SATURATION] [$job_number/$job_count] DONE $experiment_id"
  done
done

if [[ "$dry_run" == true ]]; then
  echo "[SHORT_SATURATION] Dry run only; no experiment was started."
  exit 0
fi

summary_dir="$repo_root/outputs/short_commopt_saturation/$experiment_prefix"
"$navi_python" "$repo_root/scripts/summarize_short_commopt_saturation.py" \
  --input-csv "$repo_root/outputs/navi_param.csv" \
  --experiment-prefix "$experiment_prefix" \
  --output-dir "$summary_dir"

echo "[SHORT_SATURATION] All requested jobs completed."
echo "[SHORT_SATURATION] Summary: $summary_dir"
