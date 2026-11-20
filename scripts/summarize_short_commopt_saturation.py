#!/usr/bin/env python3
"""Validate and summarize the targeted short-circuit CommOpt saturation runs."""

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


BENCHMARK_SPECS = {
    "SwapTest": {"K": 8, "spans": (5, 10)},
    "Comparator": {"K": 9, "spans": (5, 10)},
    "CuccaroAdder": {"K": 9, "spans": (5, 10)},
    "QV": {"K": 10, "spans": (5, 10)},
    "Random": {"K": 12, "spans": (5, 10, 15)},
}
FIXED_FIELDS = (
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
    "benchmark",
    "network_seed",
    "K",
    "W",
    "L_max",
    "effective_local_Lmax",
    "slice_strategy",
    "partition_assigner",
    "full_span_safeguard_enabled",
    "full_span_candidate_won",
    "telegate_cache",
    "shared_prefix_reused",
    "selected_record_count",
    "selected_epairs_internal",
    "commopt_time_sec",
    "epr_relative_l10",
    "runtime_relative_l10",
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
    return int(numeric) if math.isfinite(numeric) and numeric.is_integer() else None


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


def _single_full_span_cat_or_telegate(row: dict[str, str]) -> bool:
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


def summarize(input_csv: Path, experiment_prefix: str, output_dir: Path) -> dict[str, Path]:
    benchmark_by_slug = {name.lower(): name for name in BENCHMARK_SPECS}
    pattern = re.compile(
        rf"^{re.escape(experiment_prefix)}_(?P<benchmark>[a-z]+)_netseed(?P<seed>\d+)$"
    )
    selected: list[dict[str, str]] = []
    with input_csv.open(newline="", encoding="utf-8") as handle:
        for source in csv.DictReader(handle):
            match = pattern.fullmatch(str(source.get("experiment_id", "")))
            if match and match.group("benchmark") in benchmark_by_slug:
                row = dict(source)
                row["_seed_from_id"] = str(int(match.group("seed")))
                row["_benchmark_from_id"] = benchmark_by_slug[match.group("benchmark")]
                selected.append(row)

    errors: list[str] = []
    if not selected:
        errors.append(f"no rows found for experiment prefix {experiment_prefix!r}")

    grouped: dict[tuple[str, int], list[dict[str, str]]] = defaultdict(list)
    for row in selected:
        grouped[(row["_benchmark_from_id"], int(row["_seed_from_id"]))].append(row)

    seeds_by_benchmark: dict[str, set[int]] = defaultdict(set)
    for (benchmark, seed), rows in sorted(grouped.items()):
        seeds_by_benchmark[benchmark].add(seed)
        spec = BENCHMARK_SPECS[benchmark]
        expected = {(1, span) for span in spec["spans"]}
        counts = Counter(
            (_integer(row.get("W")), _integer(row.get("L_max"))) for row in rows
        )
        if set(counts) != expected or any(count != 1 for count in counts.values()):
            errors.append(
                f"{benchmark} network seed {seed} does not contain exactly {sorted(expected)}"
            )
        for row in rows:
            run_id = row.get("run_id", "<unknown>")
            width = _integer(row.get("W"))
            span = _integer(row.get("L_max"))
            k_value = _integer(row.get("K"))
            effective = _integer(row.get("effective_local_Lmax"))
            checks = (
                (_bool(row.get("is_valid")) is True, "row is not valid"),
                (row.get("benchmark") == benchmark, "benchmark differs from experiment ID"),
                (row.get("qpu_layout") == "4x20", "QPU layout is not 4x20"),
                (row.get("network_topology") == "all_to_all", "network is not all_to_all"),
                (_integer(row.get("network_seed")) == seed, "network seed differs from experiment ID"),
                (width == 1, "W is not fixed to 1"),
                (span in spec["spans"], "L_max is outside the targeted grid"),
                (k_value == spec["K"], f"K is not the expected value {spec['K']}"),
                (effective == min(span or 0, spec["K"]), "effective L_max is inconsistent with min(L_max, K)"),
                (row.get("partition_assigner") == "max_match", "partition assigner is not max_match"),
                (row.get("partition_assigner_name") == "Max-Match Partition Assigner", "partition assigner name is unexpected"),
                (row.get("slice_strategy") == "greedy_table", "SliceOpt strategy is not greedy_table"),
                (_bool(row.get("full_span_safeguard_enabled")) is False, "full-span safeguard is not disabled"),
                (_bool(row.get("telegate_cache")) is False, "TeleGate cache is not disabled"),
                (_bool(row.get("shared_prefix_reused")) is True, "shared prefix was not reused"),
                (_integer(row.get("telegate_cache_hits")) == 0, "strict cache hits are missing or nonzero"),
                (_integer(row.get("telegate_cache_relaxed_hits")) == 0, "relaxed cache hits are missing or nonzero"),
                ((_integer(row.get("telegate_total_calls")) or 0) > 0, "TeleGate call count is not positive"),
            )
            if span is not None and span < spec["K"]:
                checks += (
                    (_bool(row.get("full_span_candidate_won")) is False, "a full-span candidate won below K"),
                    (not _single_full_span_cat_or_telegate(row), "final plan is a single full-span CAT/TeleGate below K"),
                )
            epairs = _float(row.get("selected_epairs_internal"))
            runtime = _float(row.get("commopt_time_sec"))
            checks += (
                (epairs is not None and epairs >= 0 and epairs.is_integer(), "EPR count is not a nonnegative integer"),
                (runtime is not None and runtime > 0, "CommOpt runtime is not positive"),
            )
            for passed, message in checks:
                if not passed:
                    errors.append(f"{run_id}: {message}")

    expected_benchmarks = set(BENCHMARK_SPECS)
    if set(seeds_by_benchmark) != expected_benchmarks:
        errors.append("the set of completed benchmarks is incomplete")
    seed_sets = {tuple(sorted(values)) for values in seeds_by_benchmark.values()}
    if len(seed_sets) > 1:
        errors.append("network seed sets differ across benchmarks")

    for benchmark in BENCHMARK_SPECS:
        benchmark_rows = [row for row in selected if row["_benchmark_from_id"] == benchmark]
        for field in FIXED_FIELDS:
            values = {row.get(field) for row in benchmark_rows}
            if not values or values == {None} or values == {""}:
                errors.append(f"{benchmark}: fixed field {field} is missing")
            elif len(values) > 1:
                errors.append(f"{benchmark}: fixed field {field} differs across network seeds")

    report_path = output_dir / "validation_report.json"
    report = {
        "valid": not errors,
        "experiment_prefix": experiment_prefix,
        "network_seeds": sorted(next(iter(seed_sets), ())),
        "row_count": len(selected),
        "benchmark_specs": BENCHMARK_SPECS,
        "errors": errors,
    }
    _write_json(report_path, report)
    if errors:
        raise ValueError(
            f"short-circuit saturation validation failed; see {report_path.resolve()}"
        )

    normalized_rows: list[dict[str, Any]] = []
    for (benchmark, seed), rows in sorted(grouped.items()):
        default = next(row for row in rows if _integer(row.get("L_max")) == 10)
        default_epairs = float(default["selected_epairs_internal"])
        default_runtime = float(default["commopt_time_sec"])
        if default_epairs == 0 or default_runtime == 0:
            raise ValueError(f"{benchmark} network seed {seed} has a zero L=10 denominator")
        for source in rows:
            span = int(float(source["L_max"]))
            normalized = {
                field: source.get(field, "")
                for field in RAW_FIELDS
                if field not in {"epr_relative_l10", "runtime_relative_l10"}
            }
            normalized.update({
                "benchmark": benchmark,
                "network_seed": seed,
                "K": int(float(source["K"])),
                "W": 1,
                "L_max": span,
                "effective_local_Lmax": min(span, BENCHMARK_SPECS[benchmark]["K"]),
                "selected_record_count": int(float(source["selected_record_count"])),
                "selected_epairs_internal": float(source["selected_epairs_internal"]),
                "commopt_time_sec": float(source["commopt_time_sec"]),
                "epr_relative_l10": float(source["selected_epairs_internal"]) / default_epairs,
                "runtime_relative_l10": float(source["commopt_time_sec"]) / default_runtime,
            })
            normalized_rows.append(normalized)

    normalized_rows.sort(key=lambda row: (list(BENCHMARK_SPECS).index(row["benchmark"]), row["network_seed"], row["L_max"]))
    raw_path = output_dir / "short_saturation_raw.csv"
    _write_csv(raw_path, normalized_rows, RAW_FIELDS)

    summary_rows: list[dict[str, Any]] = []
    for benchmark, spec in BENCHMARK_SPECS.items():
        for span in spec["spans"]:
            rows = [
                row for row in normalized_rows
                if row["benchmark"] == benchmark and row["L_max"] == span
            ]
            n, epr_mean, epr_sd = _mean_sd(row["selected_epairs_internal"] for row in rows)
            _, runtime_mean, runtime_sd = _mean_sd(row["commopt_time_sec"] for row in rows)
            _, epr_relative_mean, epr_relative_sd = _mean_sd(row["epr_relative_l10"] for row in rows)
            _, runtime_relative_mean, runtime_relative_sd = _mean_sd(row["runtime_relative_l10"] for row in rows)
            summary_rows.append({
                "benchmark": benchmark,
                "K": spec["K"],
                "W": 1,
                "L_max": span,
                "effective_local_Lmax": min(span, spec["K"]),
                "n_network_seeds": n,
                "epr_mean": epr_mean,
                "epr_sample_sd": epr_sd,
                "runtime_mean_sec": runtime_mean,
                "runtime_sample_sd_sec": runtime_sd,
                "epr_relative_l10_mean": epr_relative_mean,
                "epr_relative_l10_sample_sd": epr_relative_sd,
                "runtime_relative_l10_mean": runtime_relative_mean,
                "runtime_relative_l10_sample_sd": runtime_relative_sd,
            })
    summary_path = output_dir / "short_saturation_summary.csv"
    _write_csv(summary_path, summary_rows, tuple(summary_rows[0]))

    comparison_rows: list[dict[str, Any]] = []
    for benchmark, spec in BENCHMARK_SPECS.items():
        pairs = [(5, 10, "L5_to_L10")]
        if benchmark == "Random":
            pairs.append((10, 15, "L10_to_full_span"))
        for baseline_span, candidate_span, label in pairs:
            changes = []
            for seed in sorted(seeds_by_benchmark[benchmark]):
                seed_rows = {
                    row["L_max"]: row
                    for row in normalized_rows
                    if row["benchmark"] == benchmark and row["network_seed"] == seed
                }
                baseline = seed_rows[baseline_span]
                candidate = seed_rows[candidate_span]
                changes.append({
                    "epr": candidate["selected_epairs_internal"] / baseline["selected_epairs_internal"] - 1.0,
                    "runtime": candidate["commopt_time_sec"] / baseline["commopt_time_sec"] - 1.0,
                })
            n, epr_change, epr_change_sd = _mean_sd(item["epr"] for item in changes)
            _, runtime_change, runtime_change_sd = _mean_sd(item["runtime"] for item in changes)
            comparison_rows.append({
                "benchmark": benchmark,
                "K": spec["K"],
                "comparison": label,
                "baseline_L_max": baseline_span,
                "candidate_L_max": candidate_span,
                "candidate_effective_Lmax": min(candidate_span, spec["K"]),
                "n_network_seeds": n,
                "epr_change_fraction_mean": epr_change,
                "epr_change_fraction_sample_sd": epr_change_sd,
                "runtime_change_fraction_mean": runtime_change,
                "runtime_change_fraction_sample_sd": runtime_change_sd,
            })
    comparisons_path = output_dir / "short_saturation_comparisons.csv"
    _write_csv(comparisons_path, comparison_rows, tuple(comparison_rows[0]))

    report["outputs"] = {
        "raw": str(raw_path.resolve()),
        "summary": str(summary_path.resolve()),
        "comparisons": str(comparisons_path.resolve()),
    }
    _write_json(report_path, report)
    return {
        "raw": raw_path,
        "summary": summary_path,
        "comparisons": comparisons_path,
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
    print(f"Validated short-circuit saturation runs: {outputs['validation'].resolve()}")
    print(f"Raw rows: {outputs['raw'].resolve()}")
    print(f"Parameter summary: {outputs['summary'].resolve()}")
    print(f"Pairwise comparisons: {outputs['comparisons'].resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
