#!/usr/bin/env python3
"""Build the authoritative multi-benchmark NAVI parameter dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


WIDTHS = (1, 2, 4)
SPANS = (5, 10, 15)
EXPECTED_COMBINATIONS = {(width, span) for width in WIDTHS for span in SPANS}
RUN_DIR_PATTERN = re.compile(r"^W(?P<W>\d+)_L(?P<L>\d+)$")
ANALYSIS_PREFIX = "[COMMOPT_ANALYSIS] "


def _read_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return loaded


def _read_key_values(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key.strip()] = value.strip()
    return values


def _number(value: Any) -> int | float | None:
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric.is_integer():
        return int(numeric)
    return numeric


def _parse_capacities(value: Any) -> list[int]:
    loaded = value
    if isinstance(value, str):
        try:
            loaded = json.loads(value)
        except json.JSONDecodeError:
            return []
    if not isinstance(loaded, list):
        return []
    capacities = [_number(item) for item in loaded]
    if not all(isinstance(item, (int, float)) and float(item).is_integer() for item in capacities):
        return []
    return [int(item) for item in capacities]


def _qpu_layout(qpu_count: Any, capacities: Any) -> str | None:
    count = _number(qpu_count)
    parsed = _parse_capacities(capacities)
    if not isinstance(count, (int, float)) or not float(count).is_integer() or not parsed:
        return None
    count = int(count)
    if len(parsed) == count and len(set(parsed)) == 1:
        return f"{count}x{parsed[0]}"
    return f"{count}x[{','.join(str(capacity) for capacity in parsed)}]"


def _relative_path(path: Path | None, parent: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(parent.resolve()))
    except ValueError:
        return str(path.resolve())


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _find_analysis_record(run_dir: Path) -> tuple[dict[str, Any] | None, str | None]:
    records: list[tuple[dict[str, Any], Path]] = []
    paths = sorted((*run_dir.rglob("*.txt"), *run_dir.rglob("*.log")))
    for path in paths:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.startswith(ANALYSIS_PREFIX):
                continue
            try:
                value = json.loads(line[len(ANALYSIS_PREFIX):])
            except json.JSONDecodeError as exc:
                return None, f"invalid COMMOPT_ANALYSIS JSON in {path}: {exc}"
            if isinstance(value, dict):
                records.append((value, path))
    if not records:
        return None, "COMMOPT_ANALYSIS record not found"
    unique = {json.dumps(record, sort_keys=True) for record, _ in records}
    if len(unique) != 1:
        found_paths = sorted({str(path) for _, path in records})
        return None, f"multiple distinct COMMOPT_ANALYSIS records found in {found_paths}"
    record, path = records[-1]
    record["analysis_log_path"] = path
    return record, None


def _read_navi_csv(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    task: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    if not path.exists():
        return task, metrics
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not task:
                task = {
                    "qubit_count": _number(row.get("#Qubits")),
                    "circuit_depth": _number(row.get("Depth")),
                    "gate_count": _number(row.get("#Gates")),
                    "two_qubit_gate_count": _number(row.get("#2Q Gates")),
                    "qpu_count": _number(row.get("#QPUs")),
                    "qpu_capacities": row.get("QPUs"),
                    "network_topology": row.get("net_type"),
                }
            metric_name = row.get("Metrics")
            if metric_name:
                metrics[metric_name] = _number(row.get("NAVI"))
    return task, metrics


def _extract_final_records(evaluated: dict[str, Any]) -> dict[str, Any]:
    records = evaluated.get("records", [])
    if not isinstance(records, list):
        records = []
    type_counts = Counter(str(record.get("mapping_type", "unknown")) for record in records)
    plan = []
    for record in records:
        costs = record.get("costs", {}) if isinstance(record.get("costs"), dict) else {}
        plan.append({
            "layer_start": record.get("layer_start"),
            "layer_end": record.get("layer_end"),
            "mapping_type": record.get("mapping_type"),
            "epairs": costs.get("epairs"),
            "cat_ents": costs.get("cat_ents"),
        })
    known_count = sum(type_counts.get(name, 0) for name in ("cat", "teledata", "telegate"))
    return {
        "navi_final_record_count_declared": _number(evaluated.get("num_records")),
        "navi_final_record_count": len(records),
        "navi_final_cat_record_count": type_counts.get("cat", 0),
        "navi_final_teledata_record_count": type_counts.get("teledata", 0),
        "navi_final_telegate_record_count": type_counts.get("telegate", 0),
        "navi_final_other_record_count": len(records) - known_count,
        "navi_final_record_plan": json.dumps(plan, ensure_ascii=False, separators=(",", ":")),
    }


def _discover_run_dirs(input_dirs: Iterable[Path]) -> list[tuple[Path, Path]]:
    discovered: dict[Path, Path] = {}
    for input_dir in input_dirs:
        root = input_dir.resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"input directory does not exist: {root}")
        for path in root.rglob("W*_L*"):
            if path.is_dir() and RUN_DIR_PATTERN.fullmatch(path.name):
                metadata_path = path / "run_metadata.json"
                if metadata_path.exists() and not (path / "exit_status.txt").exists():
                    try:
                        metadata = _read_json(metadata_path)
                    except (OSError, ValueError, json.JSONDecodeError):
                        metadata = None
                    if metadata is not None and metadata.get("exit_status") is None and not metadata.get("ended_at"):
                        # The runner creates metadata before launching NAVI. Do not
                        # import the currently active run during an EXIT refresh.
                        continue
                discovered.setdefault(path.resolve(), root)
    return sorted(
        ((path, source_root) for path, source_root in discovered.items()),
        key=lambda item: str(item[0]),
    )


def _collect_run(run_dir: Path, source_root: Path, path_base: Path) -> dict[str, Any]:
    match = RUN_DIR_PATTERN.fullmatch(run_dir.name)
    assert match is not None
    width = int(match.group("W"))
    span = int(match.group("L"))
    collection_errors: list[str] = []

    config_path = run_dir / "config.json"
    metadata_path = run_dir / "run_metadata.json"
    config: dict[str, Any] = {}
    metadata: dict[str, Any] = {}
    try:
        config = _read_json(config_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        collection_errors.append(f"cannot read config: {exc}")
    try:
        metadata = _read_json(metadata_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        collection_errors.append(f"cannot read metadata: {exc}")

    compiler_config = config.get("compiler_config", {}).get("navi", {})
    network_config = config.get("network_config", {})
    analysis, analysis_error = _find_analysis_record(run_dir)
    if analysis_error:
        collection_errors.append(analysis_error)
    analysis = analysis or {}

    results_path = run_dir / "results.csv"
    task, result_metrics = _read_navi_csv(results_path)
    if not results_path.exists():
        collection_errors.append("results.csv not found")

    resource = _read_key_values(run_dir / "resource_usage.txt")
    evaluated_paths = sorted(run_dir.rglob("*_NAVI_*_evaluated.json"))
    evaluated_path = evaluated_paths[-1] if evaluated_paths else None
    evaluated: dict[str, Any] = {}
    if evaluated_path is not None:
        try:
            evaluated = _read_json(evaluated_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            collection_errors.append(f"cannot read evaluated NAVI JSON: {exc}")
    else:
        collection_errors.append("evaluated NAVI JSON not found")
    evaluated_total = evaluated.get("total_costs", {}) if isinstance(evaluated.get("total_costs"), dict) else {}

    status = _number(metadata.get("exit_status"))
    if status is None:
        exit_path = run_dir / "exit_status.txt"
        status = _number(exit_path.read_text().strip()) if exit_path.exists() else None
    if status is None:
        collection_errors.append("exit status not found")

    benchmark = analysis.get("benchmark", metadata.get("benchmark", compiler_config.get("benchmark")))
    circuit_seed = analysis.get("circuit_seed")
    if circuit_seed is None:
        circuit_seed = analysis.get("random_seed", metadata.get("circuit_seed", compiler_config.get("random_seed")))
    experiment_id = metadata.get("experiment_id")
    if not experiment_id:
        experiment_id = "legacy_random" if source_root.name == "commopt_parameter_analysis" else source_root.name
    run_id = metadata.get("run_id") or f"{experiment_id}/{benchmark}/{run_dir.name}"

    final_epairs = result_metrics.get("EPR Pair Consumption")
    if final_epairs is None:
        final_epairs = _number(evaluated_total.get("epairs"))
    navi_time = result_metrics.get("Execution Time")
    if navi_time is None:
        navi_time = _number(evaluated_total.get("execution_time"))

    qpu_capacities = task.get("qpu_capacities")
    parsed_capacities = _parse_capacities(qpu_capacities)
    requested_qubit_count = _number(metadata.get("requested_qubit_count"))
    if requested_qubit_count is None and parsed_capacities:
        capacity_total = sum(parsed_capacities)
        actual_qubits = task.get("qubit_count")
        if actual_qubits in (capacity_total, capacity_total - 1):
            # Legacy parameter runs requested a circuit that filled the four
            # core capacities; SwapTest then adjusted an even request by one.
            requested_qubit_count = capacity_total

    process_wall_time = _number(resource.get("process_wall_time_sec"))
    shared_prefix_reused = bool(metadata.get("shared_prefix_reused", False))
    shared_setup_time = _number(metadata.get("shared_setup_time_sec"))
    standalone_estimated_wall_time = process_wall_time
    if (
        shared_prefix_reused
        and isinstance(process_wall_time, (int, float))
        and isinstance(shared_setup_time, (int, float))
    ):
        standalone_estimated_wall_time = process_wall_time + shared_setup_time

    row: dict[str, Any] = {
        "experiment_id": str(experiment_id),
        "replicate_id": None,
        "run_id": str(run_id),
        "is_valid": False,
        "group_complete": False,
        "validation_errors": "",
        "group_validation_errors": "",
        "started_at": metadata.get("started_at"),
        "ended_at": metadata.get("ended_at"),
        "exit_status": status,
        "git_commit": metadata.get("git_commit"),
        "git_dirty": metadata.get("git_dirty"),
        "config_sha256": metadata.get("config_sha256"),
        "config_sha256_actual": _sha256(config_path),
        "benchmark": benchmark,
        "circuit_seed": _number(circuit_seed),
        "eligibility_status": metadata.get("eligibility_status", "eligible"),
        "atomic_zero_cut_subcircuit_count": _number(
            metadata.get("atomic_zero_cut_subcircuit_count", analysis.get("K"))
        ),
        "minimum_atomic_zero_cut_subcircuits": _number(
            metadata.get(
                "minimum_atomic_zero_cut_subcircuits",
                compiler_config.get("min_atomic_zero_cut_subcircuits"),
            )
        ),
        "requested_qubit_count": requested_qubit_count,
        **task,
        "qpu_layout": _qpu_layout(task.get("qpu_count"), qpu_capacities),
        "backend": metadata.get("backend"),
        "calibration_date": metadata.get("calibration_date"),
        "network_seed": network_config.get("seed"),
        "K": analysis.get("K", metadata.get("atomic_zero_cut_subcircuit_count")),
        "slice_strategy": analysis.get(
            "slice_strategy",
            # Runs created before slice_strategy was recorded used the legacy
            # DP planner. New runs persist the resolved strategy explicitly.
            compiler_config.get("slice_strategy", "dp"),
        ),
        "W": analysis.get("W", compiler_config.get("hybrid_beam_width", width)),
        "L_max": analysis.get("L_max", compiler_config.get("hybrid_max_merge_span", span)),
        "_expected_parameter_combinations": metadata.get("expected_parameter_combinations"),
        "effective_local_Lmax": analysis.get("effective_local_Lmax"),
        "merge_probe_budget": analysis.get("merge_probe_budget", compiler_config.get("hybrid_merge_probe_budget")),
        "pos_probe_budget": analysis.get("pos_probe_budget", compiler_config.get("hybrid_pos_probe_budget")),
        "telegate_cache": compiler_config.get("telegate_cache", True),
        "telegate_cache_mode": compiler_config.get("telegate_cache_mode", "range"),
        "telegate_iteration": compiler_config.get("iteration", 50),
        "shared_prefix_reused": shared_prefix_reused,
        "shared_prefix_id": metadata.get("shared_prefix_id"),
        "shared_prefix_path": metadata.get("shared_prefix_path"),
        "shared_prefix_sha256": metadata.get("shared_prefix_sha256"),
        "shared_prefix_time_sec": _number(metadata.get("shared_prefix_time_sec")),
        "shared_setup_time_sec": shared_setup_time,
        "shared_prefix_build_wall_time_sec": _number(
            metadata.get("shared_prefix_build_wall_time_sec")
        ),
        "partition_assigner": analysis.get("partition_assigner", compiler_config.get("partition_assigner")),
        "partition_assigner_name": analysis.get("partition_assigner_name"),
        "teledata_epairs_internal": analysis.get("teledata_epairs_internal"),
        "hybrid_epairs_internal": analysis.get("hybrid_epairs_internal"),
        "selected_epairs_internal": analysis.get("selected_epairs_internal"),
        "selected_source": analysis.get("selected_source"),
        "navi_final_epairs_evaluated": final_epairs,
        "teledata_record_count": analysis.get("teledata_record_count"),
        "hybrid_record_count": analysis.get("hybrid_record_count"),
        "selected_record_count": analysis.get("selected_record_count"),
        **_extract_final_records(evaluated),
        "commopt_time_sec": analysis.get("commopt_time_sec"),
        "navi_execution_time_sec": navi_time,
        "process_wall_time_sec": process_wall_time,
        "standalone_estimated_process_wall_time_sec": standalone_estimated_wall_time,
        "peak_rss_kb": _number(resource.get("peak_rss_kb")),
        "telegate_total_calls": analysis.get("telegate_total_calls"),
        "telegate_cache_hits": analysis.get("telegate_cache_hits"),
        "telegate_cache_relaxed_hits": analysis.get("telegate_cache_relaxed_hits"),
        "telegate_cache_misses": analysis.get("telegate_cache_misses"),
        "full_span_safeguard_enabled": analysis.get(
            "full_span_safeguard_enabled",
            compiler_config.get("hybrid_full_span_safeguard", True),
        ),
        "full_span_candidate_won": analysis.get("full_span_candidate_won"),
        "config_path": _relative_path(config_path, path_base),
        "results_csv": _relative_path(results_path if results_path.exists() else None, path_base),
        "evaluated_json": _relative_path(evaluated_path, path_base),
        "analysis_log": _relative_path(analysis.get("analysis_log_path"), path_base),
        "hybrid_epairs_relative_default": None,
        "selected_epairs_relative_default": None,
        "navi_final_epairs_relative_default": None,
        "commopt_time_relative_default": None,
        "_validation_errors": collection_errors,
    }
    return row


def _is_nonnegative_integer(value: Any) -> bool:
    return isinstance(value, (int, float)) and value >= 0 and math.isclose(value, round(value))


def _append_error(row: dict[str, Any], message: str) -> None:
    errors = row.setdefault("_validation_errors", [])
    if message not in errors:
        errors.append(message)


def _validate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    for row in rows:
        prefix = row["run_id"]
        if row.get("exit_status") != 0:
            _append_error(row, f"process exit status is {row.get('exit_status')}")
        if not row.get("config_sha256") or row.get("config_sha256") != row.get("config_sha256_actual"):
            _append_error(row, "configuration SHA-256 is missing or does not match")
        if row.get("partition_assigner") != "max_match":
            _append_error(row, "partition_assigner is not max_match")
        if row.get("partition_assigner_name") != "Max-Match Partition Assigner":
            _append_error(row, f"actual partition assigner name is {row.get('partition_assigner_name')!r}")
        budget = row.get("merge_probe_budget")
        span = row.get("L_max")
        if not isinstance(budget, (int, float)) or not isinstance(span, (int, float)) or budget < span:
            _append_error(row, "merge probe budget does not cover L_max")
        for field in (
            "teledata_epairs_internal",
            "hybrid_epairs_internal",
            "selected_epairs_internal",
            "navi_final_epairs_evaluated",
        ):
            if not _is_nonnegative_integer(row.get(field)):
                _append_error(row, f"{field} is not a nonnegative integer")
        for field in ("commopt_time_sec", "navi_execution_time_sec", "process_wall_time_sec"):
            value = row.get(field)
            if not isinstance(value, (int, float)) or value <= 0:
                _append_error(row, f"{field} is not positive")
        if row.get("shared_prefix_reused"):
            for field in (
                "shared_prefix_time_sec",
                "shared_setup_time_sec",
                "shared_prefix_build_wall_time_sec",
                "standalone_estimated_process_wall_time_sec",
            ):
                value = row.get(field)
                if not isinstance(value, (int, float)) or value <= 0:
                    _append_error(row, f"{field} is not positive for a reused prefix")
            if not row.get("shared_prefix_id"):
                _append_error(row, "shared_prefix_id is missing for a reused prefix")
            if not row.get("shared_prefix_sha256"):
                _append_error(row, "shared_prefix_sha256 is missing for a reused prefix")
        declared = row.get("navi_final_record_count_declared")
        actual = row.get("navi_final_record_count")
        if declared != actual:
            _append_error(row, f"declared final record count {declared!r} differs from actual {actual!r}")

        teledata = row.get("teledata_epairs_internal")
        hybrid = row.get("hybrid_epairs_internal")
        selected = row.get("selected_epairs_internal")
        if all(isinstance(value, (int, float)) for value in (teledata, hybrid, selected)):
            expected_source = "teledata" if teledata < hybrid else "hybrid"
            expected_selected = teledata if teledata < hybrid else hybrid
            if row.get("selected_source") != expected_source:
                _append_error(row, f"safeguard selected {row.get('selected_source')!r}, expected {expected_source!r}")
            if selected != expected_selected:
                _append_error(row, "selected internal EPR is inconsistent with safeguard")

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("experiment_id")), str(row.get("benchmark")))].append(row)

    group_reports = []
    fixed_fields = (
        "benchmark",
        "circuit_seed",
        "requested_qubit_count",
        "qubit_count",
        "circuit_depth",
        "gate_count",
        "two_qubit_gate_count",
        "qpu_count",
        "qpu_capacities",
        "qpu_layout",
        "network_topology",
        "backend",
        "calibration_date",
            "K",
            "slice_strategy",
            "telegate_cache",
            "telegate_cache_mode",
            "full_span_safeguard_enabled",
            "shared_prefix_reused",
        "shared_prefix_id",
    )
    for group_key, group_rows in sorted(groups.items()):
        declared_grids = {
            json.dumps(row.get("_expected_parameter_combinations"), sort_keys=True)
            for row in group_rows
            if row.get("_expected_parameter_combinations") is not None
        }
        expected_combinations = EXPECTED_COMBINATIONS
        grid_errors = []
        if declared_grids:
            if len(declared_grids) != 1 or any(
                row.get("_expected_parameter_combinations") is None for row in group_rows
            ):
                grid_errors.append("expected parameter combinations differ across runs")
            else:
                declared_grid = group_rows[0]["_expected_parameter_combinations"]
                try:
                    expected_combinations = {
                        (int(item["W"]), int(item["L_max"])) for item in declared_grid
                    }
                    if not expected_combinations or len(expected_combinations) != len(declared_grid):
                        raise ValueError
                except (KeyError, TypeError, ValueError):
                    grid_errors.append("expected parameter combinations are invalid")
                    expected_combinations = EXPECTED_COMBINATIONS
        combination_counts = Counter(
            (int(row["W"]), int(row["L_max"]))
            for row in group_rows
            if row.get("W") is not None and row.get("L_max") is not None
        )
        found = set(combination_counts)
        structural_errors = list(grid_errors)
        group_errors = []
        missing = sorted(expected_combinations - found)
        unexpected = sorted(found - expected_combinations)
        duplicates = sorted(combo for combo, count in combination_counts.items() if count != 1)
        if missing:
            structural_errors.append(f"missing parameter combinations: {missing}")
        if unexpected:
            structural_errors.append(f"unexpected parameter combinations: {unexpected}")
        if duplicates:
            structural_errors.append(f"duplicate parameter combinations: {duplicates}")
        if len(group_rows) != len(expected_combinations):
            structural_errors.append(
                f"expected {len(expected_combinations)} rows, found {len(group_rows)}"
            )
        group_errors.extend(structural_errors)
        for field in fixed_fields:
            values = {json.dumps(row.get(field), sort_keys=True) for row in group_rows}
            if len(values) > 1:
                group_errors.append(f"fixed input field {field} differs across runs")
        if any(row.get("_validation_errors") for row in group_rows):
            group_errors.append("one or more parameter runs failed row-level validation")

        complete = not structural_errors
        for row in group_rows:
            row["group_complete"] = complete
            row["group_validation_errors"] = "; ".join(group_errors)
        group_reports.append({
            "experiment_id": group_key[0],
            "benchmark": group_key[1],
            "complete": complete,
            "found_combinations": sorted([list(item) for item in found]),
            "errors": group_errors,
        })

    for row in rows:
        row.pop("_expected_parameter_combinations", None)
        errors = row.pop("_validation_errors", [])
        row["validation_errors"] = "; ".join(errors)
        row["is_valid"] = not errors

    invalid_rows = [row["run_id"] for row in rows if not row["is_valid"]]
    incomplete_groups = [
        f"{report['experiment_id']}/{report['benchmark']}"
        for report in group_reports
        if not report["complete"]
    ]
    return {
        "valid": not invalid_rows,
        "all_groups_complete": not incomplete_groups,
        "row_count": len(rows),
        "valid_row_count": len(rows) - len(invalid_rows),
        "invalid_row_count": len(invalid_rows),
        "invalid_runs": invalid_rows,
        "group_count": len(group_reports),
        "complete_group_count": len(group_reports) - len(incomplete_groups),
        "incomplete_group_count": len(incomplete_groups),
        "incomplete_groups": incomplete_groups,
        "groups": group_reports,
        "warnings": [
            "Each parameter combination has one timing observation per experiment; filter or group by replicate_id when repeats exist.",
            "is_valid is row-level validity; use group_complete to require the experiment's declared parameter grid.",
        ],
    }


def _assign_replicate_ids(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("benchmark"),
            row.get("circuit_seed"),
            row.get("requested_qubit_count"),
            row.get("qubit_count"),
            row.get("qpu_count"),
            row.get("qpu_capacities"),
            row.get("network_topology"),
            row.get("backend"),
            row.get("calibration_date"),
            row.get("telegate_cache"),
            row.get("telegate_cache_mode"),
            row.get("W"),
            row.get("L_max"),
        )
        groups[key].append(row)
    for group_rows in groups.values():
        group_rows.sort(key=lambda row: (
            str(row.get("started_at") or ""),
            str(row.get("experiment_id") or ""),
            str(row.get("run_id") or ""),
        ))
        for replicate_id, row in enumerate(group_rows, start=1):
            row["replicate_id"] = replicate_id


def _add_relative_metrics(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row.get("experiment_id")), str(row.get("benchmark")))].append(row)
    mappings = (
        ("hybrid_epairs_internal", "hybrid_epairs_relative_default"),
        ("selected_epairs_internal", "selected_epairs_relative_default"),
        ("navi_final_epairs_evaluated", "navi_final_epairs_relative_default"),
        ("commopt_time_sec", "commopt_time_relative_default"),
    )
    for group_rows in groups.values():
        default = next(
            (row for row in group_rows if row.get("W") == 1 and row.get("L_max") == 10),
            None,
        )
        for source, destination in mappings:
            denominator = default.get(source) if default else None
            for row in group_rows:
                numerator = row.get(source)
                row[destination] = (
                    float(numerator) / float(denominator)
                    if isinstance(numerator, (int, float))
                    and isinstance(denominator, (int, float))
                    and denominator != 0
                    else None
                )


def _write_csv_atomic(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    temp_path = path.with_name(f".{path.name}.tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def _write_json_atomic(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(path)


def build_dataset(input_dirs: Iterable[Path], output_csv: Path, path_base: Path) -> dict[str, Any]:
    discovered = _discover_run_dirs(input_dirs)
    rows = [_collect_run(run_dir, source_root, path_base) for run_dir, source_root in discovered]
    rows.sort(key=lambda row: (
        str(row.get("experiment_id")),
        str(row.get("benchmark")),
        int(row.get("W") or 0),
        int(row.get("L_max") or 0),
        str(row.get("started_at") or ""),
    ))
    _assign_replicate_ids(rows)
    _add_relative_metrics(rows)
    validation = _validate(rows)
    _write_csv_atomic(rows, output_csv)

    raw_json = output_csv.with_suffix(".json")
    validation_json = output_csv.with_name(f"{output_csv.stem}.validation.json")
    _write_json_atomic(rows, raw_json)
    _write_json_atomic(validation, validation_json)
    return validation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dirs",
        nargs="*",
        type=Path,
        help="Directories recursively containing W*_L* run directories",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/navi_param.csv"),
        help="Authoritative CSV path (default: outputs/navi_param.csv)",
    )
    parser.add_argument(
        "--allow-invalid",
        action="store_true",
        help="Return success after writing even when one or more rows fail validation",
    )
    args = parser.parse_args()

    input_dirs = args.input_dirs or [
        Path("outputs/navi_param_runs"),
        Path("outputs/commopt_parameter_analysis"),
    ]
    existing_inputs = [path for path in input_dirs if path.is_dir()]
    if not existing_inputs:
        parser.error("none of the input directories exist")

    output_csv = args.output_csv.resolve()
    validation = build_dataset(existing_inputs, output_csv, Path.cwd().resolve())
    print(f"Wrote {validation['row_count']} rows to {output_csv}")
    print(f"Valid rows: {validation['valid_row_count']}; invalid rows: {validation['invalid_row_count']}")
    if validation["valid"] or args.allow_invalid:
        return 0
    print(f"Validation failed; see {output_csv.with_name(output_csv.stem + '.validation.json')}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
