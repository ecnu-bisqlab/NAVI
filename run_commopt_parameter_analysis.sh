#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
base_config="$repo_root/config.commopt_parameter_analysis.json"
navi_python="${NAVI_PYTHON:-python}"
runs_root="$repo_root/outputs/navi_param_runs"
master_csv="$repo_root/outputs/navi_param.csv"
subset_4x20_csv="$repo_root/outputs/navi_param_4x20.csv"
subset_4x50_csv="$repo_root/outputs/navi_param_4x50.csv"
subset_benchmarks="IQP,QFTAdder,PQC,MCMT,VQE"
circuit_seed=42
backend_name="ibm_marrakesh"
calibration_date="2026-03-01"

experiment_id=""
qubit_count=80
core_count=4
core_capacity=20
network_topology="all_to_all"
benchmark_spec="default"
skip_combination_spec=""
telegate_cache_override=""
telegate_cache_mode_override=""
network_seed_override=""
full_span_safeguard_override=""
slice_strategy_override=""
minimum_atomic_subcircuits_override=""
reuse_shared_prefix=true

default_benchmarks=(
  BV
  SwapTest
  Permutation
  MCMT
  Comparator
  QFTAdder
  CuccaroAdder
  PQC
  IQP
  QV
  VQE
)
all_benchmarks=(
  Random
  BV
  SwapTest
  Permutation
  MCMT
  Comparator
  QFTAdder
  CuccaroAdder
  PQC
  IQP
  QV
  VQE
)
widths=(1 2 4)
spans=(5 10 15)

usage() {
  cat <<'EOF'
Usage:
  ./run_commopt_parameter_analysis.sh [experiment_id]
  ./run_commopt_parameter_analysis.sh [options]

Options:
  --experiment-id ID       Unique output batch ID (default: current timestamp)
  --qubits N               Requested circuit qubits (default: 80)
  --cores N                Number of QPUs (default: 4)
  --capacity N             Core capacity of every QPU (default: 20)
  --network NAME           Network topology (default: all_to_all)
  --network-seed N         Network-fidelity random seed; defaults to the base
                           configuration
  --benchmarks VALUE       "all" or a comma-separated benchmark list
                           (default: the original eleven non-Random benchmarks)
  --widths VALUE           Comma-separated W values (default: 1,2,4)
  --spans VALUE            Comma-separated L_max values (default: 5,10,15)
  --skip-combinations LIST Comma-separated W:L pairs to exclude (for example 1:5)
  --telegate-cache VALUE   "on" or "off"; defaults to the base configuration
  --telegate-cache-mode M  strict, range, or range_fallback; defaults to the
                           base configuration (ignored when cache is off)
  --full-span-safeguard V  "on" or "off"; defaults to the base configuration
  --slice-strategy VALUE   "dp" or "greedy_table"; defaults to the base
                           configuration
  --min-atomic-subcircuits N
                           Minimum required atomic zero-cut subcircuits;
                           defaults to the base configuration
  --reuse-shared-prefix V  "on" or "off" (default: on). When enabled, build
                           zero-cut and Max Match once per benchmark.
  -h, --help               Show this help

Example for the 4x50 sweep:
  ./run_commopt_parameter_analysis.sh \
    --experiment-id commopt_4x50_200q \
    --qubits 200 --cores 4 --capacity 50 \
    --network all_to_all --benchmarks all

Example for the uncached 4x20 sensitivity rerun:
  ./run_commopt_parameter_analysis.sh \
    --experiment-id commopt_4x20_nocache \
    --qubits 80 --cores 4 --capacity 20 \
    --network all_to_all \
    --benchmarks IQP,QFTAdder,PQC,MCMT,VQE \
    --telegate-cache off \
    --reuse-shared-prefix on
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
    --experiment-id)
      require_value "$1" "${2:-}"
      experiment_id="$2"
      shift 2
      ;;
    --qubits)
      require_value "$1" "${2:-}"
      qubit_count="$2"
      shift 2
      ;;
    --cores)
      require_value "$1" "${2:-}"
      core_count="$2"
      shift 2
      ;;
    --capacity)
      require_value "$1" "${2:-}"
      core_capacity="$2"
      shift 2
      ;;
    --network)
      require_value "$1" "${2:-}"
      network_topology="$2"
      shift 2
      ;;
    --network-seed)
      require_value "$1" "${2:-}"
      network_seed_override="$2"
      shift 2
      ;;
    --benchmarks)
      require_value "$1" "${2:-}"
      benchmark_spec="$2"
      shift 2
      ;;
    --widths)
      require_value "$1" "${2:-}"
      IFS=',' read -r -a widths <<< "$2"
      shift 2
      ;;
    --spans)
      require_value "$1" "${2:-}"
      IFS=',' read -r -a spans <<< "$2"
      shift 2
      ;;
    --skip-combinations)
      require_value "$1" "${2:-}"
      skip_combination_spec="$2"
      shift 2
      ;;
    --telegate-cache)
      require_value "$1" "${2:-}"
      telegate_cache_override="$2"
      shift 2
      ;;
    --telegate-cache-mode)
      require_value "$1" "${2:-}"
      telegate_cache_mode_override="$2"
      shift 2
      ;;
    --full-span-safeguard)
      require_value "$1" "${2:-}"
      full_span_safeguard_override="$2"
      shift 2
      ;;
    --slice-strategy)
      require_value "$1" "${2:-}"
      slice_strategy_override="$2"
      shift 2
      ;;
    --min-atomic-subcircuits)
      require_value "$1" "${2:-}"
      minimum_atomic_subcircuits_override="$2"
      shift 2
      ;;
    --reuse-shared-prefix)
      require_value "$1" "${2:-}"
      case "$2" in
        on|true) reuse_shared_prefix=true ;;
        off|false) reuse_shared_prefix=false ;;
        *)
          echo "[ERROR] --reuse-shared-prefix must be on or off: $2" >&2
          exit 2
          ;;
      esac
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      echo "[ERROR] Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ -n "$experiment_id" ]]; then
        echo "[ERROR] Only one positional experiment_id is allowed." >&2
        usage >&2
        exit 2
      fi
      experiment_id="$1"
      shift
      ;;
  esac
done

experiment_id="${experiment_id:-$(date '+%Y%m%d_%H%M%S')}"
experiment_dir="$runs_root/$experiment_id"

for numeric_option in qubit_count core_count core_capacity; do
  value="${!numeric_option}"
  if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
    echo "[ERROR] $numeric_option must be a positive integer: $value" >&2
    exit 2
  fi
done

declare -A allowed_widths=()
declare -A allowed_spans=()
for parameter_name in widths spans; do
  declare -n parameter_values="$parameter_name"
  if (( ${#parameter_values[@]} == 0 )); then
    echo "[ERROR] --$parameter_name must not be empty." >&2
    exit 2
  fi
  declare -A seen_parameter_values=()
  for value in "${parameter_values[@]}"; do
    if [[ ! "$value" =~ ^[1-9][0-9]*$ ]]; then
      echo "[ERROR] --$parameter_name values must be positive integers: $value" >&2
      exit 2
    fi
    if [[ -n "${seen_parameter_values[$value]:-}" ]]; then
      echo "[ERROR] --$parameter_name contains a duplicate value: $value" >&2
      exit 2
    fi
    seen_parameter_values["$value"]=1
    if [[ "$parameter_name" == "widths" ]]; then
      allowed_widths["$value"]=1
    else
      allowed_spans["$value"]=1
    fi
  done
done

declare -A skipped_combinations=()
if [[ -n "$skip_combination_spec" ]]; then
  IFS=',' read -r -a skip_values <<< "$skip_combination_spec"
  for combination in "${skip_values[@]}"; do
    if [[ ! "$combination" =~ ^([1-9][0-9]*):([1-9][0-9]*)$ ]]; then
      echo "[ERROR] --skip-combinations entries must use W:L with positive integers: $combination" >&2
      exit 2
    fi
    skip_width="${BASH_REMATCH[1]}"
    skip_span="${BASH_REMATCH[2]}"
    if [[ -z "${allowed_widths[$skip_width]:-}" || -z "${allowed_spans[$skip_span]:-}" ]]; then
      echo "[ERROR] Skipped combination is outside the requested parameter grid: $combination" >&2
      exit 2
    fi
    if [[ -n "${skipped_combinations[$combination]:-}" ]]; then
      echo "[ERROR] --skip-combinations contains a duplicate entry: $combination" >&2
      exit 2
    fi
    skipped_combinations["$combination"]=1
  done
fi

if [[ "$benchmark_spec" == "default" ]]; then
  benchmarks=("${default_benchmarks[@]}")
elif [[ "$benchmark_spec" == "all" ]]; then
  benchmarks=("${all_benchmarks[@]}")
else
  IFS=',' read -r -a benchmarks <<< "$benchmark_spec"
  if (( ${#benchmarks[@]} == 0 )); then
    echo "[ERROR] --benchmarks must not be empty." >&2
    exit 2
  fi
  declare -A supported_benchmarks=()
  for benchmark in "${all_benchmarks[@]}"; do
    supported_benchmarks["$benchmark"]=1
  done
  for benchmark in "${benchmarks[@]}"; do
    if [[ -z "$benchmark" || -z "${supported_benchmarks[$benchmark]:-}" ]]; then
      echo "[ERROR] Unsupported benchmark in --benchmarks: ${benchmark:-<empty>}" >&2
      exit 2
    fi
  done
fi

if [[ ! "$experiment_id" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "[ERROR] experiment_id may contain only letters, digits, dot, underscore, and hyphen." >&2
  exit 2
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "[ERROR] jq is required to generate per-run configurations." >&2
  exit 2
fi
widths_csv="$(IFS=,; echo "${widths[*]}")"
spans_csv="$(IFS=,; echo "${spans[*]}")"
expected_parameter_combinations="$(
  jq -cn \
    --arg widths "$widths_csv" \
    --arg spans "$spans_csv" \
    --arg skips "$skip_combination_spec" \
    '($widths | split(",") | map(tonumber)) as $width_values
     | ($spans | split(",") | map(tonumber)) as $span_values
     | ($skips
        | if length == 0 then []
          else split(",")
               | map(split(":") | {W: (.[0] | tonumber), L_max: (.[1] | tonumber)})
          end) as $skipped
     | [$width_values[] as $W
        | $span_values[] as $L_max
        | select(any($skipped[]; .W == $W and .L_max == $L_max) | not)
        | {W: $W, L_max: $L_max}]'
)"
if [[ "$(jq 'length' <<< "$expected_parameter_combinations")" == "0" ]]; then
  echo "[ERROR] --skip-combinations excludes the entire requested parameter grid." >&2
  exit 2
fi
if [[ ! -x /usr/bin/time ]]; then
  echo "[ERROR] /usr/bin/time is required to collect wall time and peak RSS." >&2
  exit 2
fi
if [[ ! -f "$base_config" ]]; then
  echo "[ERROR] Base configuration not found: $base_config" >&2
  exit 2
fi
base_telegate_cache="$(
  jq -r '.compiler_config.navi.telegate_cache // true' "$base_config"
)"
case "$telegate_cache_override" in
  "") telegate_cache="$base_telegate_cache" ;;
  on|true) telegate_cache=true ;;
  off|false) telegate_cache=false ;;
  *)
    echo "[ERROR] --telegate-cache must be on or off: $telegate_cache_override" >&2
    exit 2
    ;;
esac
if [[ "$telegate_cache" != "true" && "$telegate_cache" != "false" ]]; then
  echo "[ERROR] compiler_config.navi.telegate_cache must be boolean." >&2
  exit 2
fi

base_telegate_cache_mode="$(
  jq -r '.compiler_config.navi.telegate_cache_mode // "strict"' "$base_config"
)"
telegate_cache_mode="${telegate_cache_mode_override:-$base_telegate_cache_mode}"
case "$telegate_cache_mode" in
  strict|range|range_fallback) ;;
  *)
    echo "[ERROR] --telegate-cache-mode must be strict, range, or range_fallback: $telegate_cache_mode" >&2
    exit 2
    ;;
esac

base_network_seed="$(jq -r '.network_config.seed // 42' "$base_config")"
network_seed="${network_seed_override:-$base_network_seed}"
if [[ ! "$network_seed" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "[ERROR] --network-seed must be a nonnegative integer: $network_seed" >&2
  exit 2
fi

base_full_span_safeguard="$(
  jq -r '.compiler_config.navi.hybrid_full_span_safeguard // true' "$base_config"
)"
case "$full_span_safeguard_override" in
  "") full_span_safeguard="$base_full_span_safeguard" ;;
  on|true) full_span_safeguard=true ;;
  off|false) full_span_safeguard=false ;;
  *)
    echo "[ERROR] --full-span-safeguard must be on or off: $full_span_safeguard_override" >&2
    exit 2
    ;;
esac
if [[ "$full_span_safeguard" != "true" && "$full_span_safeguard" != "false" ]]; then
  echo "[ERROR] compiler_config.navi.hybrid_full_span_safeguard must be boolean." >&2
  exit 2
fi
base_slice_strategy="$(
  jq -r '.compiler_config.navi.slice_strategy // "greedy_table"' "$base_config"
)"
slice_strategy="${slice_strategy_override:-$base_slice_strategy}"
case "$slice_strategy" in
  dp|greedy_table) ;;
  *)
    echo "[ERROR] --slice-strategy must be dp or greedy_table: $slice_strategy" >&2
    exit 2
    ;;
esac
base_minimum_atomic_zero_cut_subcircuits="$(
  jq -r '.compiler_config.navi.min_atomic_zero_cut_subcircuits // 0' "$base_config"
)"
minimum_atomic_zero_cut_subcircuits="${minimum_atomic_subcircuits_override:-$base_minimum_atomic_zero_cut_subcircuits}"
if [[ ! "$minimum_atomic_zero_cut_subcircuits" =~ ^[0-9]+$ ]]; then
  echo "[ERROR] --min-atomic-subcircuits must be a nonnegative integer: $minimum_atomic_zero_cut_subcircuits" >&2
  exit 2
fi
if ! "$navi_python" -c 'import sys, qiskit; assert sys.version_info >= (3, 10)' >/dev/null 2>&1; then
  echo "[ERROR] NAVI_PYTHON must point to Python 3.10+ with Qiskit installed: $navi_python" >&2
  exit 2
fi
if [[ -e "$experiment_dir" ]]; then
  echo "[ERROR] Experiment directory already exists; choose a new experiment_id: $experiment_dir" >&2
  exit 2
fi

mkdir -p "$experiment_dir/.matplotlib"
export MPLCONFIGDIR="$experiment_dir/.matplotlib"

git_commit="$(git -C "$repo_root" rev-parse HEAD 2>/dev/null || true)"
if [[ -n "$(git -C "$repo_root" status --porcelain 2>/dev/null)" ]]; then
  git_dirty=true
else
  git_dirty=false
fi

analysis_inputs=("$runs_root")
legacy_random="$repo_root/outputs/commopt_parameter_analysis"
if [[ -d "$legacy_random" ]]; then
  analysis_inputs+=("$legacy_random")
fi

analysis_failed=0
refresh_dataset() {
  echo "[INFO] Refreshing master parameter dataset: $master_csv"
  if ! "$navi_python" "$repo_root/scripts/analyze_commopt_parameters.py" \
    "${analysis_inputs[@]}" \
    --output-csv "$master_csv" \
    --allow-invalid; then
    analysis_failed=1
    echo "[ERROR] Failed to rebuild the master parameter dataset." >&2
    return 1
  fi
  if ! "$navi_python" "$repo_root/scripts/export_navi_parameter_subset.py" \
    --input-csv "$master_csv" \
    --output-csv "$subset_4x20_csv" \
    --qpu-layout "4x20" \
    --benchmarks "$subset_benchmarks" \
    --min-atomic-subcircuits "$base_minimum_atomic_zero_cut_subcircuits" \
    --telegate-cache false \
    --shared-prefix-reused true; then
    analysis_failed=1
    echo "[ERROR] Failed to rebuild the uncached 4x20 parameter subset." >&2
    return 1
  fi
  if ! "$navi_python" "$repo_root/scripts/export_navi_parameter_subset.py" \
    --input-csv "$master_csv" \
    --output-csv "$subset_4x50_csv" \
    --qpu-layout "4x50" \
    --benchmarks "$subset_benchmarks" \
    --min-atomic-subcircuits "$base_minimum_atomic_zero_cut_subcircuits"; then
    analysis_failed=1
    echo "[ERROR] Failed to rebuild the 4x50 parameter subset." >&2
    return 1
  fi
}

refresh_on_exit=true
final_refresh() {
  local status=$?
  trap - EXIT
  if [[ "$refresh_on_exit" == true ]]; then
    refresh_dataset || true
  fi
  exit "$status"
}
trap final_refresh EXIT

interrupted_status=0
trap 'interrupted_status=130' INT
trap 'interrupted_status=143' TERM

failed=0
cd "$repo_root"
for benchmark in "${benchmarks[@]}"; do
  benchmark_ineligible=false
  benchmark_dir="$experiment_dir/$benchmark"
  shared_prefix_args=()
  shared_prefix_id=""
  shared_prefix_relative_path=""
  shared_prefix_sha256=""
  shared_prefix_time_sec=null
  shared_setup_time_sec=null
  shared_prefix_build_wall_time_sec=null

  if [[ "$reuse_shared_prefix" == true ]]; then
    mkdir -p "$benchmark_dir"
    shared_prefix_id="${experiment_id}/${benchmark}/zero_cut_max_match"
    shared_prefix_config="$benchmark_dir/shared_prefix_config.json"
    shared_prefix_bundle="$benchmark_dir/shared_prefix.pkl"
    shared_prefix_manifest="$benchmark_dir/shared_prefix_manifest.json"
    shared_prefix_resource="$benchmark_dir/shared_prefix_resource_usage.txt"
    shared_prefix_relative_path="${experiment_id}/${benchmark}/shared_prefix.pkl"

    jq \
      --arg output_folder "$benchmark_dir/shared_prefix_outputs" \
      --argjson telegate_cache "$telegate_cache" \
      --arg telegate_cache_mode "$telegate_cache_mode" \
      --argjson network_seed "$network_seed" \
      --argjson full_span_safeguard "$full_span_safeguard" \
      --arg slice_strategy "$slice_strategy" \
      --argjson minimum_atomic_subcircuits "$minimum_atomic_zero_cut_subcircuits" \
      '.output_folder = $output_folder
       | .compiler_config.navi.telegate_cache = $telegate_cache
       | .compiler_config.navi.telegate_cache_mode = $telegate_cache_mode
       | .compiler_config.navi.hybrid_full_span_safeguard = $full_span_safeguard
       | .compiler_config.navi.slice_strategy = $slice_strategy
       | .compiler_config.navi.min_atomic_zero_cut_subcircuits = $minimum_atomic_subcircuits
       | .network_config.seed = $network_seed' \
      "$base_config" > "$shared_prefix_config"

    echo "[INFO] Building reusable zero-cut/Max Match prefix for $benchmark"
    set +e
    /usr/bin/time \
      -f 'process_wall_time_sec=%e\npeak_rss_kb=%M\nprocess_exit_status=%x' \
      -o "$shared_prefix_resource" \
      "$navi_python" "$repo_root/scripts/build_commopt_shared_prefix.py" \
        --config "$shared_prefix_config" \
        --output-bundle "$shared_prefix_bundle" \
        --output-manifest "$shared_prefix_manifest" \
        --shared-prefix-id "$shared_prefix_id" \
        --benchmark "$benchmark" \
        --qubits "$qubit_count" \
        --cores "$core_count" \
        --capacity "$core_capacity" \
        --network "$network_topology" \
        --backend "$backend_name" \
        --date "$calibration_date" \
        --circuit-seed "$circuit_seed" \
        > "$benchmark_dir/shared_prefix_stdout.log" \
        2> "$benchmark_dir/shared_prefix_stderr.log"
    shared_prefix_status=$?
    set -e

    if (( shared_prefix_status != 0 )); then
      failed=1
      if (( shared_prefix_status == 78 )); then
        echo "[INFO] Skipping $benchmark because its shared prefix is ineligible." >&2
      else
        echo "[ERROR] Failed to build shared prefix for $benchmark (status $shared_prefix_status)." >&2
      fi
      if (( interrupted_status != 0 )); then
        exit "$interrupted_status"
      fi
      continue
    fi

    shared_prefix_sha256="$(jq -r '.bundle_sha256' "$shared_prefix_manifest")"
    shared_prefix_time_sec="$(jq -r '.shared_prefix_time_sec' "$shared_prefix_manifest")"
    shared_setup_time_sec="$(jq -r '.shared_setup_time_sec' "$shared_prefix_manifest")"
    shared_prefix_build_wall_time_sec="$(awk -F= '$1 == "process_wall_time_sec" {print $2}' "$shared_prefix_resource")"
    shared_prefix_args=(--navi_shared_prefix "$shared_prefix_bundle")
  fi

  for width in "${widths[@]}"; do
    for span in "${spans[@]}"; do
      if [[ -n "${skipped_combinations["$width:$span"]:-}" ]]; then
        echo "[INFO] Skipping excluded parameter combination W=$width, L_max=$span"
        continue
      fi
      short_run_id="W${width}_L${span}"
      run_id="${experiment_id}/${benchmark}/${short_run_id}"
      run_dir="$experiment_dir/$benchmark/$short_run_id"
      mkdir -p "$run_dir"
      run_config="$run_dir/config.json"

      jq \
        --arg output_folder "$run_dir" \
        --argjson width "$width" \
        --argjson span "$span" \
        --argjson telegate_cache "$telegate_cache" \
        --arg telegate_cache_mode "$telegate_cache_mode" \
        --argjson network_seed "$network_seed" \
        --argjson full_span_safeguard "$full_span_safeguard" \
        --arg slice_strategy "$slice_strategy" \
        --argjson minimum_atomic_subcircuits "$minimum_atomic_zero_cut_subcircuits" \
        '.output_folder = $output_folder
         | .compiler_config.navi.hybrid_beam_width = $width
         | .compiler_config.navi.hybrid_max_merge_span = $span
         | .compiler_config.navi.telegate_cache = $telegate_cache
         | .compiler_config.navi.telegate_cache_mode = $telegate_cache_mode
         | .compiler_config.navi.hybrid_full_span_safeguard = $full_span_safeguard
         | .compiler_config.navi.slice_strategy = $slice_strategy
         | .compiler_config.navi.min_atomic_zero_cut_subcircuits = $minimum_atomic_subcircuits
         | .network_config.seed = $network_seed' \
        "$base_config" > "$run_config"

      config_sha256="$(sha256sum "$run_config" | awk '{print $1}')"
      started_at="$(date --iso-8601=seconds)"
      jq -n \
        --arg experiment_id "$experiment_id" \
        --arg run_id "$run_id" \
        --arg started_at "$started_at" \
        --arg git_commit "$git_commit" \
        --arg config_sha256 "$config_sha256" \
        --arg benchmark "$benchmark" \
        --arg backend "$backend_name" \
        --arg calibration_date "$calibration_date" \
        --arg network_topology "$network_topology" \
        --argjson git_dirty "$git_dirty" \
        --argjson circuit_seed "$circuit_seed" \
        --argjson requested_qubit_count "$qubit_count" \
        --argjson requested_qpu_count "$core_count" \
        --argjson requested_qpu_capacity "$core_capacity" \
        --argjson expected_parameter_combinations "$expected_parameter_combinations" \
        --argjson telegate_cache "$telegate_cache" \
        --arg telegate_cache_mode "$telegate_cache_mode" \
        --argjson network_seed "$network_seed" \
        --argjson full_span_safeguard "$full_span_safeguard" \
        --arg slice_strategy "$slice_strategy" \
        --argjson minimum_atomic_subcircuits "$minimum_atomic_zero_cut_subcircuits" \
        --argjson shared_prefix_reused "$reuse_shared_prefix" \
        --arg shared_prefix_id "$shared_prefix_id" \
        --arg shared_prefix_path "$shared_prefix_relative_path" \
        --arg shared_prefix_sha256 "$shared_prefix_sha256" \
        --argjson shared_prefix_time_sec "$shared_prefix_time_sec" \
        --argjson shared_setup_time_sec "$shared_setup_time_sec" \
        --argjson shared_prefix_build_wall_time_sec "$shared_prefix_build_wall_time_sec" \
        --argjson W "$width" \
        --argjson L_max "$span" \
        '{experiment_id: $experiment_id, run_id: $run_id,
          started_at: $started_at, git_commit: $git_commit,
          git_dirty: $git_dirty, config_sha256: $config_sha256,
          benchmark: $benchmark, circuit_seed: $circuit_seed,
          requested_qubit_count: $requested_qubit_count,
          requested_qpu_count: $requested_qpu_count,
          requested_qpu_capacities: [range(0; $requested_qpu_count) | $requested_qpu_capacity],
          expected_parameter_combinations: $expected_parameter_combinations,
          telegate_cache: $telegate_cache,
          telegate_cache_mode: $telegate_cache_mode,
          network_seed: $network_seed,
          full_span_safeguard_enabled: $full_span_safeguard,
          slice_strategy: $slice_strategy,
          minimum_atomic_zero_cut_subcircuits: $minimum_atomic_subcircuits,
          shared_prefix_reused: $shared_prefix_reused,
          shared_prefix_id: (if $shared_prefix_reused then $shared_prefix_id else null end),
          shared_prefix_path: (if $shared_prefix_reused then $shared_prefix_path else null end),
          shared_prefix_sha256: (if $shared_prefix_reused then $shared_prefix_sha256 else null end),
          shared_prefix_time_sec: $shared_prefix_time_sec,
          shared_setup_time_sec: $shared_setup_time_sec,
          shared_prefix_build_wall_time_sec: $shared_prefix_build_wall_time_sec,
          network_topology: $network_topology,
          backend: $backend, calibration_date: $calibration_date,
          W: $W, L_max: $L_max}' > "$run_dir/run_metadata.json"

      echo "[INFO] Running $run_id (${qubit_count}q, ${core_count}x${core_capacity}, ${network_topology}, network_seed=${network_seed}, min_atomic_subcircuits=${minimum_atomic_zero_cut_subcircuits}, full_span_safeguard=${full_span_safeguard}, telegate_cache=${telegate_cache}, cache_mode=${telegate_cache_mode}, shared_prefix=${reuse_shared_prefix})"
      set +e
      /usr/bin/time \
        -f 'process_wall_time_sec=%e\npeak_rss_kb=%M\nprocess_exit_status=%x' \
        -o "$run_dir/resource_usage.txt" \
        "$navi_python" main.py \
          -gconf "$run_config" \
          -cname "$benchmark" \
          -nq "$qubit_count" \
          -core "$core_count" \
          -cap "$core_capacity" \
          -net "$network_topology" \
          -bname "$backend_name" \
          -date "$calibration_date" \
          -seed "$circuit_seed" \
          -ocsv results.csv \
          "${shared_prefix_args[@]}" \
          > "$run_dir/launcher_stdout.log" \
          2> "$run_dir/launcher_stderr.log"
      status=$?
      set -e

      ineligible_line="$(
        rg --no-filename -m 1 '^\[COMMOPT_INELIGIBLE\] ' "$run_dir" \
          -g '*.txt' -g '*.log' 2>/dev/null | head -n 1 || true
      )"
      ineligible_json=""
      if [[ -n "$ineligible_line" ]]; then
        benchmark_ineligible=true
        ineligible_json="${ineligible_line#\[COMMOPT_INELIGIBLE\] }"
        status=78
      fi

      printf '%s\n' "$status" > "$run_dir/exit_status.txt"
      ended_at="$(date --iso-8601=seconds)"
      tmp_metadata="$run_dir/run_metadata.tmp.json"
      jq \
        --arg ended_at "$ended_at" \
        --argjson exit_status "$status" \
        '. + {ended_at: $ended_at, exit_status: $exit_status}' \
        "$run_dir/run_metadata.json" > "$tmp_metadata"
      mv "$tmp_metadata" "$run_dir/run_metadata.json"

      if [[ "$benchmark_ineligible" == true ]]; then
        atomic_subcircuit_count="$(
          jq -r '.atomic_zero_cut_subcircuit_count' <<< "$ineligible_json"
        )"
        minimum_required="$(jq -r '.minimum_required' <<< "$ineligible_json")"
        jq \
          --arg eligibility_status "ineligible" \
          --argjson atomic_zero_cut_subcircuit_count "$atomic_subcircuit_count" \
          --argjson minimum_atomic_zero_cut_subcircuits "$minimum_required" \
          '. + {
            eligibility_status: $eligibility_status,
            atomic_zero_cut_subcircuit_count: $atomic_zero_cut_subcircuit_count,
            minimum_atomic_zero_cut_subcircuits: $minimum_atomic_zero_cut_subcircuits
          }' \
          "$run_dir/run_metadata.json" > "$tmp_metadata"
        mv "$tmp_metadata" "$run_dir/run_metadata.json"
      fi

      if (( status != 0 )); then
        failed=1
        echo "[ERROR] $run_id failed with exit status $status" >&2
      fi

      refresh_dataset || failed=1

      if (( interrupted_status != 0 )); then
        echo "[INFO] Interrupted; completed runs are available in $master_csv" >&2
        exit "$interrupted_status"
      fi
      if [[ "$benchmark_ineligible" == true ]]; then
        echo "[INFO] Stopping $benchmark: only $atomic_subcircuit_count atomic zero-cut subcircuits; minimum is $minimum_required."
        break 2
      fi
    done
  done
done

refresh_on_exit=false
if (( analysis_failed != 0 )); then
  failed=1
fi

validation_report="$repo_root/outputs/navi_param.validation.json"
expected_group_count="${#benchmarks[@]}"
if [[ ! -f "$validation_report" ]] || ! jq -e \
  --arg experiment_id "$experiment_id" \
  --arg run_prefix "$experiment_id/" \
  --argjson expected_group_count "$expected_group_count" \
  '([.invalid_runs[] | select(startswith($run_prefix))] | length) == 0
   and ([.groups[] | select(.experiment_id == $experiment_id)] | length) == $expected_group_count
   and ([.groups[]
         | select(.experiment_id == $experiment_id)
         | select((.complete | not) or (.errors | length > 0))]
        | length) == 0' \
  "$validation_report" >/dev/null; then
  failed=1
  echo "[ERROR] Current experiment is incomplete or contains invalid runs; see $validation_report" >&2
fi

echo "[INFO] Master parameter dataset: $master_csv"
exit "$failed"
