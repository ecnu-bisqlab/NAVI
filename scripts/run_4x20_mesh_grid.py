#!/usr/bin/env python3
"""Run and manage the audited 4x20 mesh-grid five-compiler experiment."""

from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_TEMPLATE = REPO_ROOT / "config.4x20_mesh_grid_greedy_w1_l10.json"
AUDIT_ROOT = REPO_ROOT / "outputs" / "4x20_mesh_grid_greedy_w1_l10"
DEFAULT_EXPERIMENT_ID = "260901"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "outputs" / "260901-4x20.csv"
DEFAULT_PYTHON = Path(sys.executable)
ACTIVE_FILE = AUDIT_ROOT / "active.json"

BENCHMARKS = (
    "BV",
    "SwapTest",
    "Permutation",
    "MCMT",
    "Comparator",
    "QFTAdder",
    "CuccaroAdder",
    "PQC",
    "IQP",
    "Random",
    "QV",
    "VQE",
)
METRICS = (
    "F_norm",
    "Cumulative Error",
    "EPR Pair Consumption",
    "Execution Time",
)
METHODS = ("Static OEE", "FGP-rOEE", "AutoComm", "WBCP", "NAVI")
CSV_COLUMNS = (
    "Circuit",
    "#Qubits",
    "#Gates",
    "#2Q Gates",
    "Depth",
    "#QPUs",
    "QPUs",
    "net_type",
    "Metrics",
    *METHODS,
)
CONFIG_EXPECTATIONS = {
    "ablation_mode": False,
    "slice_strategy": "greedy_table",
    "partition_assigner": "max_match",
    "hybrid_beam_width": 1,
    "hybrid_max_merge_span": 10,
    "hybrid_merge_probe_budget": 15,
    "hybrid_full_span_safeguard": True,
    "multi_record_mapper": "boundeddp_neighbor",
    "beam_width": 1,
    "mapopt_dominance_safeguard": True,
    "min_atomic_zero_cut_subcircuits": 0,
    "telegate_cache": False,
}


class ExperimentError(RuntimeError):
    pass


def _now() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExperimentError(f"JSON root must be an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_benchmarks(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(BENCHMARKS)
    requested = [item.strip() for item in value.split(",") if item.strip()]
    if not requested:
        raise ExperimentError("at least one benchmark is required")
    unknown = [item for item in requested if item not in BENCHMARKS]
    if unknown:
        raise ExperimentError(
            f"unknown benchmarks {unknown}; available={list(BENCHMARKS)}"
        )
    if len(set(requested)) != len(requested):
        raise ExperimentError("benchmark list contains duplicates")
    return requested


def validate_config(config: dict[str, Any]) -> None:
    if config.get("compiler_modules") != ["src.baselines", "src.navi"]:
        raise ExperimentError("config must register src.baselines and src.navi")
    expected_ids = ["staticoee", "fgproee", "wbcp", "autocomm", "navi"]
    if config.get("compiler_ids") != expected_ids:
        raise ExperimentError(f"compiler_ids must equal {expected_ids}")
    if bool(config.get("navi_checkpoint", {}).get("enabled", True)):
        raise ExperimentError("NAVI checkpoints must be disabled")
    navi = config.get("compiler_config", {}).get("navi", {})
    for key, expected in CONFIG_EXPECTATIONS.items():
        if navi.get(key) != expected:
            raise ExperimentError(
                f"compiler_config.navi.{key}={navi.get(key)!r}, expected {expected!r}"
            )
    network = config.get("network_config", {})
    expected_network = {
        "fidelity_range": [0.90, 0.93],
        "comm_slot_reserve": 2,
        "seed": 42,
    }
    for key, expected in expected_network.items():
        if network.get(key) != expected:
            raise ExperimentError(
                f"network_config.{key}={network.get(key)!r}, expected {expected!r}"
            )


def validate_attempt_csv(path: Path, benchmark: str) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ExperimentError(f"result CSV is missing or empty: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != CSV_COLUMNS:
            raise ExperimentError(
                f"unexpected CSV columns in {path}: {reader.fieldnames!r}"
            )
        rows = list(reader)
    if len(rows) != len(METRICS):
        raise ExperimentError(
            f"{benchmark} must contain {len(METRICS)} rows, found {len(rows)}"
        )
    if [row["Metrics"] for row in rows] != list(METRICS):
        raise ExperimentError(
            f"{benchmark} metric order differs from {list(METRICS)}"
        )
    for row in rows:
        if row["Circuit"] != benchmark:
            raise ExperimentError(
                f"unexpected circuit {row['Circuit']!r} in {benchmark} result"
            )
        if int(row["#QPUs"]) != 4:
            raise ExperimentError(f"{benchmark} did not use four QPUs")
        try:
            capacities = ast.literal_eval(row["QPUs"])
        except (SyntaxError, ValueError) as exc:
            raise ExperimentError(f"invalid QPU capacity list for {benchmark}") from exc
        if capacities != [20, 20, 20, 20]:
            raise ExperimentError(
                f"{benchmark} QPU capacities are {capacities!r}, expected four 20s"
            )
        if row["net_type"] != "mesh_grid":
            raise ExperimentError(
                f"{benchmark} topology is {row['net_type']!r}, expected mesh_grid"
            )
        for method in METHODS:
            raw = row.get(method, "")
            try:
                value = float(raw)
            except (TypeError, ValueError) as exc:
                raise ExperimentError(
                    f"{benchmark} has no numeric {method} value for {row['Metrics']}"
                ) from exc
            if not math.isfinite(value):
                raise ExperimentError(
                    f"{benchmark} has non-finite {method} value for {row['Metrics']}"
                )
    return rows


def _find_json_record(log_path: Path, prefix: str) -> dict[str, Any]:
    matches: list[dict[str, Any]] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(prefix):
            try:
                payload = json.loads(line[len(prefix) :])
            except json.JSONDecodeError as exc:
                raise ExperimentError(f"invalid audit JSON in {log_path}: {exc}") from exc
            if isinstance(payload, dict):
                matches.append(payload)
    if not matches:
        raise ExperimentError(f"audit record {prefix.strip()} not found in {log_path}")
    return matches[-1]


def validate_navi_audit(
    log_path: Path,
    selection_path: Path,
    runtime_config: Path,
) -> dict[str, Any]:
    config = _load_json(runtime_config)
    validate_config(config)
    analysis = _find_json_record(log_path, "[COMMOPT_ANALYSIS] ")
    expected_analysis = {
        "slice_strategy": "greedy_table",
        "W": 1,
        "L_max": 10,
        "full_span_safeguard_enabled": True,
    }
    for key, expected in expected_analysis.items():
        if analysis.get(key) != expected:
            raise ExperimentError(
                f"CommOpt audit {key}={analysis.get(key)!r}, expected {expected!r}"
            )
    selection = _load_json(selection_path)
    if selection.get("mapper_id") == "direct":
        raise ExperimentError("MapOpt unexpectedly used the direct mapper on mesh_grid")
    if selection.get("safeguard_enabled") is not True:
        raise ExperimentError("MapOpt dominance safeguard was not enabled")
    return {"commopt": analysis, "mapopt": selection}


def check_target_ownership(
    output_csv: Path,
    manifest_path: Path,
    config_sha256: str,
) -> None:
    if not output_csv.exists() or output_csv.stat().st_size == 0:
        return
    if not manifest_path.is_file():
        raise ExperimentError(
            f"refusing to overwrite non-empty unowned result file: {output_csv}"
        )
    manifest = _load_json(manifest_path)
    if Path(str(manifest.get("output_csv", ""))).resolve() != output_csv.resolve():
        raise ExperimentError("manifest does not own the requested output CSV")
    if manifest.get("config_sha256") != config_sha256:
        raise ExperimentError("manifest configuration hash differs from the current config")


def rebuild_master_csv(
    output_csv: Path,
    completed_csvs: dict[str, Path],
) -> int:
    rows: list[dict[str, str]] = []
    for benchmark in BENCHMARKS:
        source = completed_csvs.get(benchmark)
        if source is not None:
            rows.extend(validate_attempt_csv(source, benchmark))
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_csv.with_name(output_csv.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, output_csv)
    return len(rows)


def _batch_paths(experiment_id: str) -> tuple[Path, Path, Path]:
    batch_dir = AUDIT_ROOT / experiment_id
    return batch_dir, batch_dir / "manifest.json", batch_dir / "runner.log"


def _runtime_config(config: dict[str, Any], batch_dir: Path) -> dict[str, Any]:
    result = json.loads(json.dumps(config))
    result["output_folder"] = str((batch_dir / "artifacts").resolve())
    return result


def _load_or_create_manifest(
    experiment_id: str,
    output_csv: Path,
    config_sha256: str,
) -> tuple[dict[str, Any], Path, Path]:
    batch_dir, manifest_path, runner_log = _batch_paths(experiment_id)
    batch_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        manifest = _load_json(manifest_path)
        if manifest.get("experiment_id") != experiment_id:
            raise ExperimentError("manifest experiment ID mismatch")
        if Path(str(manifest.get("output_csv", ""))).resolve() != output_csv.resolve():
            raise ExperimentError("manifest output CSV mismatch")
        if manifest.get("config_sha256") != config_sha256:
            raise ExperimentError("manifest configuration hash mismatch")
    else:
        manifest = {
            "format_version": 1,
            "experiment_id": experiment_id,
            "created_at": _now(),
            "updated_at": _now(),
            "config_template": str(CONFIG_TEMPLATE),
            "config_sha256": config_sha256,
            "output_csv": str(output_csv.resolve()),
            "circuit_seed_policy": "benchmark_defaults",
            "benchmarks": {},
        }
        _atomic_json(manifest_path, manifest)
    return manifest, manifest_path, runner_log


def _completed_sources(manifest: dict[str, Any]) -> dict[str, Path]:
    completed: dict[str, Path] = {}
    states = manifest.get("benchmarks", {})
    if not isinstance(states, dict):
        raise ExperimentError("manifest benchmarks field must be an object")
    for benchmark, state in states.items():
        if isinstance(state, dict) and state.get("status") == "completed":
            completed[benchmark] = Path(str(state["results_csv"]))
    return completed


def _event(runner_log: Path, message: str) -> None:
    rendered = f"[{_now()}] {message}"
    print(rendered, flush=True)
    runner_log.parent.mkdir(parents=True, exist_ok=True)
    with runner_log.open("a", encoding="utf-8") as handle:
        handle.write(rendered + "\n")


def _next_attempt_dir(batch_dir: Path, benchmark: str) -> Path:
    parent = batch_dir / "attempts" / benchmark
    parent.mkdir(parents=True, exist_ok=True)
    numbers = []
    for candidate in parent.glob("attempt_*"):
        try:
            numbers.append(int(candidate.name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue
    attempt = parent / f"attempt_{max(numbers, default=0) + 1:03d}"
    attempt.mkdir()
    return attempt


def _new_files(root: Path, pattern: str, before: set[Path]) -> list[Path]:
    return sorted(
        (path for path in root.rglob(pattern) if path.resolve() not in before),
        key=lambda path: path.stat().st_mtime,
    )


def _run_benchmark(
    python_bin: Path,
    benchmark: str,
    batch_dir: Path,
    runtime_config_path: Path,
    attempt_dir: Path,
) -> tuple[dict[str, Any], str | None]:
    artifacts = batch_dir / "artifacts"
    before_logs = {path.resolve() for path in artifacts.rglob("*.txt")} if artifacts.exists() else set()
    before_selections = (
        {path.resolve() for path in artifacts.rglob("*_mapopt_selection.json")}
        if artifacts.exists()
        else set()
    )
    results_csv = attempt_dir / "results.csv"
    launcher_log = attempt_dir / "launcher.log"
    command = [
        str(python_bin),
        str(REPO_ROOT / "main.py"),
        "-gconf",
        str(runtime_config_path),
        "-cname",
        benchmark,
        "-nq",
        "80",
        "-core",
        "4",
        "-cap",
        "20",
        "-net",
        "mesh_grid",
        "-bname",
        "ibm_marrakesh",
        "-date",
        "2026-03-01",
        "-ocsv",
        str(results_csv.resolve()),
    ]
    metadata: dict[str, Any] = {
        "benchmark": benchmark,
        "started_at": _now(),
        "command": command,
        "circuit_seed": None,
        "results_csv": str(results_csv.resolve()),
        "launcher_log": str(launcher_log.resolve()),
    }
    _atomic_json(attempt_dir / "attempt.json", metadata)
    matplotlib_config = batch_dir / ".matplotlib"
    matplotlib_config.mkdir(parents=True, exist_ok=True)
    child_environment = os.environ.copy()
    child_environment["MPLCONFIGDIR"] = str(matplotlib_config.resolve())
    child_environment["PYTHONUNBUFFERED"] = "1"
    started = time.monotonic()
    with launcher_log.open("w", encoding="utf-8") as log_handle:
        process = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            env=child_environment,
        )
    metadata["ended_at"] = _now()
    metadata["wall_time_sec"] = time.monotonic() - started
    metadata["exit_status"] = process.returncode
    try:
        if process.returncode != 0:
            raise ExperimentError(f"main.py exited with status {process.returncode}")
        validate_attempt_csv(results_csv, benchmark)
        new_logs = _new_files(artifacts, "*.txt", before_logs)
        new_selections = _new_files(
            artifacts, "*_mapopt_selection.json", before_selections
        )
        if not new_logs:
            raise ExperimentError("main.py did not create an internal experiment log")
        if not new_selections:
            raise ExperimentError("NAVI did not create a MapOpt selection audit")
        internal_log = new_logs[-1]
        selection_path = new_selections[-1]
        audit = validate_navi_audit(
            internal_log,
            selection_path,
            runtime_config_path,
        )
        metadata.update(
            {
                "status": "completed",
                "internal_log": str(internal_log.resolve()),
                "mapopt_selection": str(selection_path.resolve()),
                "audit": audit,
            }
        )
        error = None
    except ExperimentError as exc:
        metadata["status"] = "failed"
        metadata["error"] = str(exc)
        error = str(exc)
    _atomic_json(attempt_dir / "attempt.json", metadata)
    return metadata, error


def _preflight(python_bin: Path, benchmarks: Iterable[str]) -> dict[str, Any]:
    if not CONFIG_TEMPLATE.is_file():
        raise ExperimentError(f"config template not found: {CONFIG_TEMPLATE}")
    config = _load_json(CONFIG_TEMPLATE)
    validate_config(config)
    if not python_bin.is_file() or not os.access(python_bin, os.X_OK):
        raise ExperimentError(f"Python executable is unavailable: {python_bin}")
    probe = subprocess.run(
        [
            str(python_bin),
            "-c",
            "import sys,qiskit; assert sys.version_info >= (3,10); print(qiskit.__version__)",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if probe.returncode != 0:
        raise ExperimentError(f"Python/Qiskit preflight failed: {probe.stderr.strip()}")
    gurobi_probe = subprocess.run(
        [
            str(python_bin),
            "-c",
            (
                "import gurobipy as gp; "
                "model=gp.Model('navi_license_preflight'); "
                "model.dispose(); "
                "print('.'.join(map(str, gp.gurobi.version())))"
            ),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    if gurobi_probe.returncode != 0:
        detail = (gurobi_probe.stderr or gurobi_probe.stdout).strip()
        raise ExperimentError(
            "Gurobi license preflight failed; run this experiment from the "
            f"license-enabled host/session: {detail}"
        )
    backend = (
        REPO_ROOT
        / "data/device_properties/ibm/ibm_marrakesh_sampled_22q"
        / "ibm_marrakesh_sampled_22q_2026-03-01.xlsx"
    )
    if not backend.is_file():
        raise ExperimentError(f"sampled backend data not found: {backend}")
    return {
        "python": str(python_bin.resolve()),
        "qiskit": probe.stdout.strip(),
        "gurobi": gurobi_probe.stdout.strip().splitlines()[-1],
        "config": str(CONFIG_TEMPLATE),
        "config_sha256": _sha256(CONFIG_TEMPLATE),
        "backend": str(backend),
        "benchmarks": list(benchmarks),
    }


def execute_suite(args: argparse.Namespace) -> int:
    benchmarks = _parse_benchmarks(args.benchmarks)
    python_bin = Path(args.python).resolve()
    output_csv = Path(args.output_csv).resolve()
    preflight = _preflight(python_bin, benchmarks)
    batch_dir, manifest_path, runner_log = _batch_paths(args.experiment_id)
    check_target_ownership(output_csv, manifest_path, preflight["config_sha256"])
    manifest, manifest_path, runner_log = _load_or_create_manifest(
        args.experiment_id,
        output_csv,
        preflight["config_sha256"],
    )
    runtime_config_path = batch_dir / "runtime_config.json"
    runtime_config = _runtime_config(_load_json(CONFIG_TEMPLATE), batch_dir)
    _atomic_json(runtime_config_path, runtime_config)

    lock_path = batch_dir / "run.lock"
    lock_handle = lock_path.open("w", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise ExperimentError(f"experiment {args.experiment_id} is already running") from exc

    completed = _completed_sources(manifest)
    if completed:
        rebuild_master_csv(output_csv, completed)
    failures = 0
    _event(
        runner_log,
        f"START experiment={args.experiment_id} benchmarks={','.join(benchmarks)}",
    )
    try:
        for index, benchmark in enumerate(benchmarks, start=1):
            existing = completed.get(benchmark)
            if existing is not None:
                try:
                    validate_attempt_csv(existing, benchmark)
                    _event(
                        runner_log,
                        f"[{index}/{len(benchmarks)}] SKIP completed benchmark={benchmark}",
                    )
                    continue
                except ExperimentError as exc:
                    _event(
                        runner_log,
                        f"[{index}/{len(benchmarks)}] INVALIDATE benchmark={benchmark}: {exc}",
                    )

            attempt_dir = _next_attempt_dir(batch_dir, benchmark)
            _event(
                runner_log,
                f"[{index}/{len(benchmarks)}] START benchmark={benchmark} attempt={attempt_dir.name}",
            )
            metadata, error = _run_benchmark(
                python_bin,
                benchmark,
                batch_dir,
                runtime_config_path,
                attempt_dir,
            )
            manifest.setdefault("benchmarks", {})[benchmark] = {
                "status": metadata["status"],
                "attempt": str(attempt_dir.resolve()),
                "results_csv": metadata["results_csv"],
                "wall_time_sec": metadata["wall_time_sec"],
                "updated_at": _now(),
            }
            if error is not None:
                manifest["benchmarks"][benchmark]["error"] = error
                failures += 1
                _event(
                    runner_log,
                    f"[{index}/{len(benchmarks)}] FAILED benchmark={benchmark}: {error}",
                )
            else:
                completed[benchmark] = Path(metadata["results_csv"])
                row_count = rebuild_master_csv(output_csv, completed)
                _event(
                    runner_log,
                    f"[{index}/{len(benchmarks)}] COMPLETE benchmark={benchmark} "
                    f"wall={metadata['wall_time_sec']:.2f}s master_rows={row_count}",
                )
            manifest["updated_at"] = _now()
            _atomic_json(manifest_path, manifest)
    finally:
        fcntl.flock(lock_handle, fcntl.LOCK_UN)
        lock_handle.close()

    completed_count = len(_completed_sources(manifest))
    manifest["updated_at"] = _now()
    manifest["last_run_finished_at"] = _now()
    _atomic_json(manifest_path, manifest)
    _event(
        runner_log,
        f"FINISH completed={completed_count}/{len(BENCHMARKS)} failures={failures} "
        f"output={output_csv}",
    )
    return 1 if failures else 0


def _process_matches(pid: int) -> bool:
    try:
        command = (Path("/proc") / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ")
    except OSError:
        return False
    return b"run_4x20_mesh_grid.py" in command and b"_worker" in command


def _active() -> dict[str, Any] | None:
    if not ACTIVE_FILE.is_file():
        return None
    try:
        active = _load_json(ACTIVE_FILE)
        pid = int(active["pid"])
    except (ExperimentError, KeyError, TypeError, ValueError):
        return None
    return active if _process_matches(pid) else None


def _run_lock_is_held(batch_dir: Path) -> bool:
    lock_path = batch_dir / "run.lock"
    if not lock_path.exists():
        return False
    handle = lock_path.open("a", encoding="utf-8")
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(handle, fcntl.LOCK_UN)
        return False
    finally:
        handle.close()


def _remove_active_if_owned(pid: int) -> None:
    if not ACTIVE_FILE.is_file():
        return
    try:
        active = _load_json(ACTIVE_FILE)
    except ExperimentError:
        return
    if int(active.get("pid", -1)) == pid:
        ACTIVE_FILE.unlink(missing_ok=True)


def start_background(args: argparse.Namespace) -> int:
    active = _active()
    if active is not None:
        raise ExperimentError(
            f"a suite is already running: PID {active['pid']} experiment={active['experiment_id']}"
        )
    benchmarks = _parse_benchmarks(args.benchmarks)
    python_bin = Path(args.python).resolve()
    output_csv = Path(args.output_csv).resolve()
    preflight = _preflight(python_bin, benchmarks)
    batch_dir, manifest_path, runner_log = _batch_paths(args.experiment_id)
    check_target_ownership(output_csv, manifest_path, preflight["config_sha256"])
    _load_or_create_manifest(
        args.experiment_id,
        output_csv,
        preflight["config_sha256"],
    )
    batch_dir.mkdir(parents=True, exist_ok=True)
    background_log = batch_dir / "background.log"
    command = [
        str(python_bin),
        str(Path(__file__).resolve()),
        "_worker",
        "--experiment-id",
        args.experiment_id,
        "--benchmarks",
        args.benchmarks,
        "--python",
        str(python_bin),
        "--output-csv",
        str(output_csv),
    ]
    with background_log.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    active_payload = {
        "pid": process.pid,
        "experiment_id": args.experiment_id,
        "started_at": _now(),
        "background_log": str(background_log.resolve()),
        "runner_log": str(runner_log.resolve()),
        "output_csv": str(output_csv),
    }
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    _atomic_json(ACTIVE_FILE, active_payload)
    print(f"Started experiment {args.experiment_id} (PID {process.pid}).")
    print(f"Log: {background_log}")
    print(f"CSV: {output_csv}")
    return 0


def worker(args: argparse.Namespace) -> int:
    try:
        return execute_suite(args)
    finally:
        _remove_active_if_owned(os.getpid())


def show_status(args: argparse.Namespace) -> int:
    batch_dir, manifest_path, _ = _batch_paths(args.experiment_id)
    active = _active()
    if active is not None and active.get("experiment_id") == args.experiment_id:
        state = f"RUNNING pid={active['pid']}"
    elif _run_lock_is_held(batch_dir):
        state = "RUNNING foreground/managed worker"
    else:
        state = "NOT RUNNING"
    completed: list[str] = []
    failed: list[str] = []
    if manifest_path.is_file():
        manifest = _load_json(manifest_path)
        for benchmark, item in manifest.get("benchmarks", {}).items():
            if item.get("status") == "completed":
                completed.append(benchmark)
            elif item.get("status") == "failed":
                failed.append(benchmark)
    print(state)
    print(f"Experiment: {args.experiment_id}")
    print(f"Completed ({len(completed)}/{len(BENCHMARKS)}): {','.join(completed) or '-'}")
    print(f"Failed: {','.join(failed) or '-'}")
    print(f"Audit: {batch_dir}")
    print(f"CSV: {args.output_csv}")
    return 0


def _tail(path: Path, line_count: int) -> None:
    if not path.is_file():
        raise ExperimentError(f"log file does not exist: {path}")
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    print("\n".join(lines[-line_count:]))


def show_log(args: argparse.Namespace) -> int:
    batch_dir, _, runner_log = _batch_paths(args.experiment_id)
    background_log = batch_dir / "background.log"
    active = _active()
    path = (
        background_log
        if active is not None
        and active.get("experiment_id") == args.experiment_id
        and background_log.is_file()
        else runner_log
    )
    _tail(path, args.lines)
    return 0


def follow_log(args: argparse.Namespace) -> int:
    batch_dir, _, runner_log = _batch_paths(args.experiment_id)
    background_log = batch_dir / "background.log"
    active = _active()
    path = (
        background_log
        if active is not None
        and active.get("experiment_id") == args.experiment_id
        and background_log.exists()
        else runner_log
    )
    if not path.exists():
        raise ExperimentError(f"log file does not exist: {path}")
    return subprocess.run(["tail", "-n", "80", "-F", str(path)], check=False).returncode


def stop_background(args: argparse.Namespace) -> int:
    active = _active()
    if active is None or active.get("experiment_id") != args.experiment_id:
        print(f"No matching background worker for experiment {args.experiment_id}.")
        return 0
    pid = int(active["pid"])
    if not _process_matches(pid):
        raise ExperimentError(f"PID {pid} no longer belongs to this experiment")
    os.killpg(pid, signal.SIGTERM)
    print(f"Sent SIGTERM to experiment process group {pid}; completed results are resumable.")
    return 0


def validate_command(args: argparse.Namespace) -> int:
    benchmarks = _parse_benchmarks(args.benchmarks)
    preflight = _preflight(Path(args.python).resolve(), benchmarks)
    batch_dir, manifest_path, _ = _batch_paths(args.experiment_id)
    check_target_ownership(
        Path(args.output_csv).resolve(),
        manifest_path,
        preflight["config_sha256"],
    )
    print(json.dumps(preflight, ensure_ascii=False, indent=2))
    print("Commands:")
    for benchmark in benchmarks:
        print(
            f"  {args.python} main.py -gconf <runtime_config> -cname {benchmark} "
            "-nq 80 -core 4 -cap 20 -net mesh_grid "
            "-bname ibm_marrakesh -date 2026-03-01 -ocsv <attempt.csv>"
        )
    print(f"Audit directory: {batch_dir}")
    print(f"Master CSV: {Path(args.output_csv).resolve()}")
    return 0


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
    parser.add_argument("--benchmarks", default="all")
    parser.add_argument(
        "--python",
        default=os.environ.get("NAVI_4X20_PYTHON", str(DEFAULT_PYTHON)),
    )
    parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate")
    _add_run_arguments(validate_parser)
    run_parser = subparsers.add_parser("run")
    _add_run_arguments(run_parser)
    start_parser = subparsers.add_parser("start")
    _add_run_arguments(start_parser)
    worker_parser = subparsers.add_parser("_worker", help=argparse.SUPPRESS)
    _add_run_arguments(worker_parser)
    for name in ("status", "log", "follow", "stop"):
        command_parser = subparsers.add_parser(name)
        command_parser.add_argument("--experiment-id", default=DEFAULT_EXPERIMENT_ID)
        command_parser.add_argument("--output-csv", default=str(DEFAULT_OUTPUT_CSV))
        if name == "log":
            command_parser.add_argument("--lines", type=int, default=80)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    handlers = {
        "validate": validate_command,
        "run": execute_suite,
        "start": start_background,
        "_worker": worker,
        "status": show_status,
        "log": show_log,
        "follow": follow_log,
        "stop": stop_background,
    }
    try:
        return handlers[args.command](args)
    except ExperimentError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
