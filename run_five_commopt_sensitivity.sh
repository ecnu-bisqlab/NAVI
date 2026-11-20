#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
runner="$repo_root/run_commopt_parameter_analysis.sh"
base_config="$repo_root/config.commopt_parameter_analysis.json"
validation_report="$repo_root/outputs/navi_param.validation.json"
navi_python="${NAVI_PYTHON:-python}"
seeds_spec="42,43,44"
experiment_prefix="commopt5_nofs"
dry_run=false

# Fastest first, based on the completed uncached 4x20 measurements.
benchmarks=(VQE MCMT PQC QFTAdder IQP)

usage() {
  cat <<'EOF'
Usage:
  ./run_five_commopt_sensitivity.sh [options]

Options:
  --seeds LIST              Comma-separated network seeds (default: 42,43,44)
  --experiment-prefix NAME  Experiment prefix (default: commopt5_nofs)
  --dry-run                 Print the ordered jobs without running experiments
  -h, --help                Show this help

Execution order is benchmark-major and sequential:
  VQE:       seed42 -> seed43 -> seed44
  MCMT:      seed42 -> seed43 -> seed44
  PQC:       seed42 -> seed43 -> seed44
  QFTAdder:  seed42 -> seed43 -> seed44
  IQP:       seed42 -> seed43 -> seed44

Every job evaluates W={1,2,4}, L_max={5,10,15} with greedy-table SliceOpt,
max_match, the full-span safeguard disabled, the TeleGate cache disabled, and
shared-prefix reuse enabled. Existing complete jobs are skipped; incomplete
directories are never overwritten.
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
  echo "[ERROR] jq is required to validate the fixed sensitivity configuration." >&2
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
  echo "[ERROR] Five-benchmark sensitivity requires greedy_table, partition_assigner=max_match, and Telegate seed 26." >&2
  exit 2
fi
if ! "$navi_python" -c 'import sys; assert sys.version_info >= (3, 10)' >/dev/null 2>&1; then
  echo "[ERROR] NAVI_PYTHON must point to Python 3.10+." >&2
  exit 2
fi

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
  echo "[FIVE_SENSITIVITY] Benchmark: $benchmark"
  for seed in "${seeds[@]}"; do
    job_number=$((job_number + 1))
    experiment_id="$(experiment_id_for "$benchmark" "$seed")"
    if [[ -n "${skip_jobs[$experiment_id]:-}" ]]; then
      echo "[FIVE_SENSITIVITY] [$job_number/$job_count] SKIP complete $experiment_id"
      continue
    fi
    if [[ "$dry_run" == true ]]; then
      echo "[FIVE_SENSITIVITY] [$job_number/$job_count] DRY-RUN $benchmark network_seed=$seed experiment_id=$experiment_id"
      continue
    fi

    echo "[FIVE_SENSITIVITY] [$job_number/$job_count] START $benchmark network_seed=$seed experiment_id=$experiment_id"
    NAVI_PYTHON="$navi_python" "$runner" \
      --experiment-id "$experiment_id" \
      --qubits 80 \
      --cores 4 \
      --capacity 20 \
      --network all_to_all \
      --network-seed "$seed" \
      --benchmarks "$benchmark" \
      --widths 1,2,4 \
      --spans 5,10,15 \
      --telegate-cache off \
      --full-span-safeguard off \
      --reuse-shared-prefix on
    echo "[FIVE_SENSITIVITY] [$job_number/$job_count] DONE $experiment_id"
  done

  if [[ "$dry_run" == false && "$benchmark" == "VQE" ]]; then
    vqe_prefix="${experiment_prefix}_vqe"
    vqe_summary_dir="$repo_root/outputs/vqe_commopt_sensitivity/$vqe_prefix"
    "$navi_python" "$repo_root/scripts/summarize_vqe_commopt_sensitivity.py" \
      --input-csv "$repo_root/outputs/navi_param.csv" \
      --experiment-prefix "$vqe_prefix" \
      --output-dir "$vqe_summary_dir"
    echo "[FIVE_SENSITIVITY] VQE summary: $vqe_summary_dir"
  fi
done

if [[ "$dry_run" == true ]]; then
  echo "[FIVE_SENSITIVITY] Dry run only; no experiment was started."
else
  echo "[FIVE_SENSITIVITY] All requested jobs completed."
  echo "[FIVE_SENSITIVITY] Master data: $repo_root/outputs/navi_param.csv"
fi
