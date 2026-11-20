#!/usr/bin/env python3
"""Validate and summarize fixed-circuit VQE CommOpt sensitivity runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


WIDTHS = (1, 2, 4)
SPANS = (5, 10, 15)
EXPECTED_GRID = {(width, span) for width in WIDTHS for span in SPANS}
FIXED_CIRCUIT_FIELDS = (
    "circuit_seed",
    "qubit_count",
    "circuit_depth",
    "gate_count",
    "two_qubit_gate_count",
    "qpu_count",
    "qpu_capacities",
    "backend",
    "calibration_date",
    "telegate_cache_mode",
    "telegate_iteration",
    "K",
)
RAW_FIELDS = (
    "experiment_id",
    "run_id",
    "network_seed",
    "benchmark",
    "W",
    "L_max",
    "K",
    "slice_strategy",
    "partition_assigner",
    "full_span_safeguard_enabled",
    "full_span_candidate_won",
    "telegate_cache",
    "shared_prefix_reused",
    "selected_record_count",
    "selected_epairs_internal",
    "commopt_time_sec",
    "telegate_total_calls",
    "telegate_cache_hits",
    "telegate_cache_relaxed_hits",
    "epr_relative_default",
    "runtime_relative_default",
)


def _bool(value: Any) -> bool | None:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    return None


def _integer(value: Any) -> int | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return int(numeric) if numeric.is_integer() else None


def _float(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _mean_sd(values: Iterable[float]) -> tuple[int, float | None, float | None]:
    collected = [float(value) for value in values]
    if not collected:
        return 0, None, None
    return (
        len(collected),
        statistics.mean(collected),
        statistics.stdev(collected) if len(collected) >= 2 else None,
    )


def _sensitivity(rows: list[dict[str, Any]], field: str) -> tuple[float | None, float | None, float | None]:
    values = {
        (int(row["W"]), int(row["L_max"])): float(row[field])
        for row in rows
    }
    if set(values) != EXPECTED_GRID:
        return None, None, None
    all_values = list(values.values())
    grand_mean = statistics.mean(all_values)
    total_variance = statistics.mean(
        (value - grand_mean) ** 2 for value in all_values
    )
    if math.isclose(total_variance, 0.0, abs_tol=1e-15):
        return None, None, None
    width_means = [
        statistics.mean(values[(width, span)] for span in SPANS)
        for width in WIDTHS
    ]
    span_means = [
        statistics.mean(values[(width, span)] for width in WIDTHS)
        for span in SPANS
    ]
    width_effect = statistics.mean(
        (value - grand_mean) ** 2 for value in width_means
    ) / total_variance
    span_effect = statistics.mean(
        (value - grand_mean) ** 2 for value in span_means
    ) / total_variance
    interaction = 1.0 - width_effect - span_effect
    if math.isclose(interaction, 0.0, abs_tol=1e-12):
        interaction = 0.0
    return width_effect, span_effect, interaction


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _is_single_full_span_plan(row: dict[str, str]) -> bool:
    try:
        plan = json.loads(row.get("navi_final_record_plan", ""))
    except (TypeError, json.JSONDecodeError):
        return False
    return (
        isinstance(plan, list)
        and len(plan) == 1
        and isinstance(plan[0], dict)
        and str(plan[0].get("mapping_type", "")).lower() in {"cat", "telegate"}
    )


def summarize(
    input_csv: Path,
    experiment_prefix: str,
    output_dir: Path,
) -> dict[str, Path]:
    pattern = re.compile(rf"^{re.escape(experiment_prefix)}_netseed(?P<seed>\d+)$")
    with input_csv.open(newline="", encoding="utf-8") as handle:
        selected = []
        for source in csv.DictReader(handle):
            match = pattern.fullmatch(str(source.get("experiment_id", "")))
            if match:
                row = dict(source)
                row["_seed_from_id"] = int(match.group("seed"))
                selected.append(row)

    errors: list[str] = []
    if not selected:
        errors.append(
            f"no rows found for experiment prefix {experiment_prefix!r}"
        )

    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        grouped[int(row["_seed_from_id"])].append(row)

    for seed, rows in sorted(grouped.items()):
        combinations = Counter(
            (_integer(row.get("W")), _integer(row.get("L_max")))
            for row in rows
        )
        if set(combinations) != EXPECTED_GRID or any(count != 1 for count in combinations.values()):
            errors.append(
                f"network seed {seed} does not contain exactly one row for each parameter point"
            )
        for row in rows:
            run_id = row.get("run_id", "<unknown>")
            checks = (
                (_bool(row.get("is_valid")) is True, "row is not valid"),
                (row.get("benchmark") == "VQE", "benchmark is not VQE"),
                (row.get("qpu_layout") == "4x20", "QPU layout is not 4x20"),
                (row.get("network_topology") == "all_to_all", "network is not all_to_all"),
                (_integer(row.get("network_seed")) == seed, "network seed differs from experiment ID"),
                (row.get("partition_assigner") == "max_match", "partition assigner is not max_match"),
                (row.get("partition_assigner_name") == "Max-Match Partition Assigner", "partition assigner name is unexpected"),
                (row.get("slice_strategy") == "greedy_table", "SliceOpt strategy is not greedy_table"),
                (_bool(row.get("full_span_safeguard_enabled")) is False, "full-span safeguard is not disabled"),
                (_bool(row.get("full_span_candidate_won")) is False, "full-span candidate won"),
                (_bool(row.get("telegate_cache")) is False, "TeleGate cache is not disabled"),
                (_bool(row.get("shared_prefix_reused")) is True, "shared prefix was not reused"),
                ((_integer(row.get("K")) or 0) > max(SPANS), "K does not exceed the largest tested L_max"),
                ((_integer(row.get("L_max")) or 0) < (_integer(row.get("K")) or 0), "L_max is not smaller than K"),
                (_integer(row.get("selected_record_count")) is not None and int(float(row["selected_record_count"])) > 1, "selected plan contains at most one record"),
                (not _is_single_full_span_plan(row), "final plan is a single full-span CAT/TeleGate record"),
                ((_integer(row.get("telegate_total_calls")) or 0) > 0, "TeleGate call count is not positive"),
                (_integer(row.get("telegate_cache_hits")) == 0, "strict cache hits are missing or nonzero"),
                (_integer(row.get("telegate_cache_relaxed_hits")) == 0, "relaxed cache hits are missing or nonzero"),
            )
            epairs = _float(row.get("selected_epairs_internal"))
            runtime = _float(row.get("commopt_time_sec"))
            checks += (
                (epairs is not None and epairs >= 0 and float(epairs).is_integer(), "EPR count is not a nonnegative integer"),
                (runtime is not None and runtime > 0, "CommOpt runtime is not positive"),
            )
            for passed, message in checks:
                if not passed:
                    errors.append(f"{run_id}: {message}")

    for field in FIXED_CIRCUIT_FIELDS:
        values = {row.get(field) for row in selected}
        if values == {None} or values == {""}:
            errors.append(f"fixed experiment field {field} is missing")
            continue
        if len(values) > 1:
            errors.append(f"fixed experiment field {field} differs across network seeds")

    report_path = output_dir / "validation_report.json"
    report = {
        "valid": not errors,
        "experiment_prefix": experiment_prefix,
        "network_seeds": sorted(grouped),
        "row_count": len(selected),
        "expected_parameter_grid": [list(item) for item in sorted(EXPECTED_GRID)],
        "errors": errors,
    }
    _write_json(report_path, report)
    if errors:
        raise ValueError(
            f"VQE sensitivity validation failed; see {report_path.resolve()}"
        )

    normalized_rows: list[dict[str, Any]] = []
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for seed, rows in sorted(grouped.items()):
        default = next(
            row
            for row in rows
            if _integer(row.get("W")) == 1 and _integer(row.get("L_max")) == 10
        )
        default_epairs = float(default["selected_epairs_internal"])
        default_runtime = float(default["commopt_time_sec"])
        if default_epairs == 0 or default_runtime == 0:
            raise ValueError(f"network seed {seed} has a zero default denominator")
        for source in rows:
            normalized = {
                field: source.get(field, "")
                for field in RAW_FIELDS
                if field not in {"epr_relative_default", "runtime_relative_default"}
            }
            normalized.update({
                "network_seed": seed,
                "W": int(float(source["W"])),
                "L_max": int(float(source["L_max"])),
                "K": int(float(source["K"])),
                "selected_record_count": int(float(source["selected_record_count"])),
                "selected_epairs_internal": float(source["selected_epairs_internal"]),
                "commopt_time_sec": float(source["commopt_time_sec"]),
                "epr_relative_default": float(source["selected_epairs_internal"]) / default_epairs,
                "runtime_relative_default": float(source["commopt_time_sec"]) / default_runtime,
            })
            normalized_rows.append(normalized)
            by_seed[seed].append(normalized)

    normalized_rows.sort(key=lambda row: (row["network_seed"], row["W"], row["L_max"]))
    raw_path = output_dir / "vqe_sensitivity_raw.csv"
    _write_csv(raw_path, normalized_rows, RAW_FIELDS)

    summary_rows: list[dict[str, Any]] = []
    for width, span in sorted(EXPECTED_GRID):
        point_rows = [
            row for row in normalized_rows
            if row["W"] == width and row["L_max"] == span
        ]
        _, epr_mean, epr_sd = _mean_sd(row["selected_epairs_internal"] for row in point_rows)
        _, time_mean, time_sd = _mean_sd(row["commopt_time_sec"] for row in point_rows)
        n, epr_rel_mean, epr_rel_sd = _mean_sd(row["epr_relative_default"] for row in point_rows)
        _, time_rel_mean, time_rel_sd = _mean_sd(row["runtime_relative_default"] for row in point_rows)
        summary_rows.append({
            "W": width,
            "L_max": span,
            "n_network_seeds": n,
            "epr_mean": epr_mean,
            "epr_sample_sd": epr_sd,
            "runtime_mean_sec": time_mean,
            "runtime_sample_sd_sec": time_sd,
            "epr_relative_mean": epr_rel_mean,
            "epr_relative_sample_sd": epr_rel_sd,
            "runtime_relative_mean": time_rel_mean,
            "runtime_relative_sample_sd": time_rel_sd,
        })
    summary_path = output_dir / "vqe_sensitivity_summary.csv"
    summary_fields = tuple(summary_rows[0])
    _write_csv(summary_path, summary_rows, summary_fields)

    index_rows: list[dict[str, Any]] = []
    per_metric: dict[str, list[tuple[float | None, float | None, float | None]]] = defaultdict(list)
    for seed, rows in sorted(by_seed.items()):
        for metric, field in (
            ("EPR", "epr_relative_default"),
            ("runtime", "runtime_relative_default"),
        ):
            indices = _sensitivity(rows, field)
            per_metric[metric].append(indices)
            index_rows.append({
                "row_type": "network_seed",
                "network_seed": seed,
                "metric": metric,
                "n_network_seeds": 1,
                "S_W": indices[0],
                "S_L_max": indices[1],
                "S_interaction": indices[2],
                "S_W_mean": "",
                "S_W_sample_sd": "",
                "S_L_max_mean": "",
                "S_L_max_sample_sd": "",
                "S_interaction_mean": "",
                "S_interaction_sample_sd": "",
            })
    for metric in ("EPR", "runtime"):
        indices = per_metric[metric]
        width_values = [value[0] for value in indices if value[0] is not None]
        span_values = [value[1] for value in indices if value[1] is not None]
        interaction_values = [value[2] for value in indices if value[2] is not None]
        n, width_mean, width_sd = _mean_sd(width_values)
        _, span_mean, span_sd = _mean_sd(span_values)
        _, interaction_mean, interaction_sd = _mean_sd(interaction_values)
        index_rows.append({
            "row_type": "across_network_seeds",
            "network_seed": "",
            "metric": metric,
            "n_network_seeds": n,
            "S_W": "",
            "S_L_max": "",
            "S_interaction": "",
            "S_W_mean": width_mean,
            "S_W_sample_sd": width_sd,
            "S_L_max_mean": span_mean,
            "S_L_max_sample_sd": span_sd,
            "S_interaction_mean": interaction_mean,
            "S_interaction_sample_sd": interaction_sd,
        })
    indices_path = output_dir / "vqe_sensitivity_indices.csv"
    index_fields = tuple(index_rows[0])
    _write_csv(indices_path, index_rows, index_fields)

    report["outputs"] = {
        "raw": str(raw_path.resolve()),
        "summary": str(summary_path.resolve()),
        "sensitivity_indices": str(indices_path.resolve()),
    }
    _write_json(report_path, report)
    return {
        "raw": raw_path,
        "summary": summary_path,
        "indices": indices_path,
        "validation": report_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("outputs/navi_param.csv"),
        help="Master NAVI parameter CSV",
    )
    parser.add_argument("--experiment-prefix", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.experiment_prefix):
        parser.error("--experiment-prefix contains unsupported characters")
    try:
        outputs = summarize(args.input_csv, args.experiment_prefix, args.output_dir)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"[ERROR] {exc}\n")
    print(f"Validated VQE sensitivity runs: {outputs['validation'].resolve()}")
    print(f"Raw rows: {outputs['raw'].resolve()}")
    print(f"Parameter summary: {outputs['summary'].resolve()}")
    print(f"Sensitivity indices: {outputs['indices'].resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
