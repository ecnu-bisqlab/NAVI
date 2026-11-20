#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${REPO_DIR}/outputs/navi_static_ablation_uniform_k_legacy_cat_no_fullspan"
PID_FILE="${OUTPUT_DIR}/background.pid"
LOG_FILE="${OUTPUT_DIR}/background.log"
PYTHON_BIN="${NAVI_ABLATION_PYTHON:-python3}"
RUN_SCRIPT="${SCRIPT_DIR}/run_navi_static_ablation.py"

usage() {
    printf '%s\n' \
        "Usage:" \
        "  $0 start [QV|all|priority|Circuit1,Circuit2]" \
        "  $0 status" \
        "  $0 log [lines]" \
        "  $0 follow" \
        "  $0 stop"
}

read_pid() {
    if [[ -f "${PID_FILE}" ]]; then
        tr -d '[:space:]' < "${PID_FILE}"
    fi
}

process_matches() {
    local process_id="$1"
    local process_args
    process_args="$(ps -p "${process_id}" -o args= 2>/dev/null || true)"
    [[ "${process_args}" == *"${RUN_SCRIPT}"*"--static-segmentation-mode uniform_k"* ]]
}

running_pid() {
    local process_id
    process_id="$(read_pid)"
    if [[ -n "${process_id}" ]] \
        && kill -0 "${process_id}" 2>/dev/null \
        && process_matches "${process_id}"; then
        printf '%s\n' "${process_id}"
    fi
}

start_run() {
    local benchmarks="${1:-QV}"
    local process_id

    mkdir -p "${OUTPUT_DIR}"
    process_id="$(running_pid)"
    if [[ -n "${process_id}" ]]; then
        printf 'Already running (PID %s).\n' "${process_id}"
        exit 1
    fi
    if [[ ! -x "${PYTHON_BIN}" ]]; then
        printf 'Python executable is unavailable: %s\n' "${PYTHON_BIN}" >&2
        exit 1
    fi

    printf '\n[%s] START uniform_k benchmarks=%s\n' "$(date '+%F %T %z')" "${benchmarks}" >> "${LOG_FILE}"
    cd "${REPO_DIR}"
    nohup env PYTHONUNBUFFERED=1 "${PYTHON_BIN}" "${RUN_SCRIPT}" \
        --benchmarks "${benchmarks}" \
        --static-segmentation-mode uniform_k \
        --detail-csv "outputs/navi_static_ablation_uniform_k_legacy_cat_no_fullspan/variant_results.csv" \
        --record-dir "outputs/navi_static_ablation_uniform_k_legacy_cat_no_fullspan/records" \
        </dev/null >> "${LOG_FILE}" 2>&1 &
    process_id="$!"
    printf '%s\n' "${process_id}" > "${PID_FILE}"
    printf 'Started K-matched uniform w/o SliceOpt probe in background.\n'
    printf 'PID: %s\nLog: %s\n' "${process_id}" "${LOG_FILE}"
}

show_status() {
    local process_id
    process_id="$(running_pid)"
    if [[ -n "${process_id}" ]]; then
        printf 'RUNNING (PID %s)\n' "${process_id}"
        ps -p "${process_id}" -o pid=,etime=,pcpu=,pmem=,stat=,args=
        exit 0
    fi
    printf 'NOT RUNNING\n'
    if [[ -f "${PID_FILE}" ]]; then
        printf 'The PID file may be stale or outside the current PID namespace: %s\n' "${PID_FILE}"
    fi
}

show_log() {
    local lines="${1:-80}"
    if [[ ! "${lines}" =~ ^[0-9]+$ ]]; then
        printf 'Log line count must be a nonnegative integer.\n' >&2
        exit 1
    fi
    if [[ ! -f "${LOG_FILE}" ]]; then
        printf 'No log yet: %s\n' "${LOG_FILE}"
        exit 1
    fi
    tail -n "${lines}" "${LOG_FILE}"
}

follow_log() {
    mkdir -p "${OUTPUT_DIR}"
    touch "${LOG_FILE}"
    tail -n 80 -f "${LOG_FILE}"
}

stop_run() {
    local process_id
    process_id="$(running_pid)"
    if [[ -z "${process_id}" ]]; then
        printf 'No matching uniform-K ablation process is visible.\n'
        exit 0
    fi
    kill "${process_id}"
    printf 'Sent SIGTERM to PID %s. Completed circuit results remain resumable.\n' "${process_id}"
}

command_name="${1:-}"
case "${command_name}" in
    start)
        start_run "${2:-QV}"
        ;;
    status)
        show_status
        ;;
    log)
        show_log "${2:-80}"
        ;;
    follow)
        follow_log
        ;;
    stop)
        stop_run
        ;;
    *)
        usage
        exit 1
        ;;
esac
