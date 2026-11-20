#!/usr/bin/env python3
"""Export a canonical, merged NAVI parameter-analysis subset."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


EXPECTED_COMBINATIONS = {
    (width, span)
    for width in (1, 2, 4)
    for span in (5, 10, 15)
}
MERGED_FIELDS = (
    "merged_group_id",
    "merged_group_complete",
    "merged_hybrid_epairs_relative_default",
    "merged_selected_epairs_relative_default",
    "merged_navi_final_epairs_relative_default",
    "merged_commopt_time_relative_default",
)
GROUP_FIELDS = (
    "qpu_layout",
    "benchmark",
    "circuit_seed",
    "requested_qubit_count",
    "qubit_count",
    "qpu_count",
    "qpu_capacities",
    "network_topology",
    "backend",
    "calibration_date",
    "telegate_cache",
    "telegate_cache_mode",
    "shared_prefix_reused",
)
RELATIVE_FIELDS = (
    ("hybrid_epairs_internal", "merged_hybrid_epairs_relative_default"),
    ("selected_epairs_internal", "merged_selected_epairs_relative_default"),
    ("navi_final_epairs_evaluated", "merged_navi_final_epairs_relative_default"),
    ("commopt_time_sec", "merged_commopt_time_relative_default"),
)


def _is_true(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _integer(value: Any) -> int | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return int(numeric) if numeric.is_integer() else None


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _group_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in GROUP_FIELDS)


def _group_id(key: tuple[str, ...]) -> str:
    return "|".join(f"{field}={value}" for field, value in zip(GROUP_FIELDS, key))


def _canonical_sort_key(row: dict[str, str]) -> tuple[str, str, str]:
    return (
        str(row.get("started_at", "")),
        str(row.get("experiment_id", "")),
        str(row.get("run_id", "")),
    )


def _write_csv_atomic(
    rows: Iterable[dict[str, Any]],
    fieldnames: list[str],
    output_csv: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_csv.with_name(f".{output_csv.name}.tmp")
    with temp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(output_csv)


def export_subset(
    input_csv: Path,
    output_csv: Path,
    *,
    qpu_layout: str,
    benchmarks: Iterable[str],
    minimum_atomic_subcircuits: int = 0,
    telegate_cache: bool | None = None,
    shared_prefix_reused: bool | None = None,
) -> list[dict[str, Any]]:
    benchmark_order = list(dict.fromkeys(benchmarks))
    benchmark_set = set(benchmark_order)
    with input_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        source_fields = list(reader.fieldnames or [])
        candidates = [
            dict(row)
            for row in reader
            if row.get("qpu_layout") == qpu_layout
            and row.get("benchmark") in benchmark_set
            and _is_true(row.get("is_valid"))
            and (_integer(row.get("K")) or 0) >= minimum_atomic_subcircuits
            and (
                telegate_cache is None
                or _is_true(row.get("telegate_cache")) == telegate_cache
            )
            and (
                shared_prefix_reused is None
                or _is_true(row.get("shared_prefix_reused")) == shared_prefix_reused
            )
        ]

    canonical: dict[tuple[tuple[str, ...], int, int], dict[str, str]] = {}
    for row in sorted(candidates, key=_canonical_sort_key):
        width = _integer(row.get("W"))
        span = _integer(row.get("L_max"))
        if width is None or span is None:
            continue
        canonical.setdefault((_group_key(row), width, span), row)

    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for (group_key, _, _), row in canonical.items():
        grouped[group_key].append(row)

    benchmark_rank = {benchmark: index for index, benchmark in enumerate(benchmark_order)}
    output_rows: list[dict[str, Any]] = []
    for group_key, group_rows in grouped.items():
        found = {
            (_integer(row.get("W")), _integer(row.get("L_max")))
            for row in group_rows
        }
        complete = found == EXPECTED_COMBINATIONS
        default = next(
            (
                row
                for row in group_rows
                if _integer(row.get("W")) == 1
                and _integer(row.get("L_max")) == 10
            ),
            None,
        )
        for row in group_rows:
            exported: dict[str, Any] = dict(row)
            exported["merged_group_id"] = _group_id(group_key)
            exported["merged_group_complete"] = complete
            for source, destination in RELATIVE_FIELDS:
                numerator = _float(row.get(source))
                denominator = _float(default.get(source)) if default else None
                exported[destination] = (
                    numerator / denominator
                    if numerator is not None
                    and denominator is not None
                    and denominator != 0
                    else None
                )
            output_rows.append(exported)

    output_rows.sort(key=lambda row: (
        benchmark_rank.get(str(row.get("benchmark")), len(benchmark_rank)),
        str(row.get("benchmark", "")),
        str(row.get("merged_group_id", "")),
        _integer(row.get("W")) or 0,
        _integer(row.get("L_max")) or 0,
    ))
    output_fields = source_fields + [
        field for field in MERGED_FIELDS if field not in source_fields
    ]
    _write_csv_atomic(output_rows, output_fields, output_csv)
    return output_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--qpu-layout", required=True)
    parser.add_argument(
        "--min-atomic-subcircuits",
        type=int,
        default=0,
        help="Exclude rows whose K is below this threshold",
    )
    parser.add_argument(
        "--benchmarks",
        required=True,
        help="Comma-separated benchmark names",
    )
    parser.add_argument(
        "--telegate-cache",
        choices=("true", "false"),
        help="Optionally keep only runs with this TeleGate cache setting",
    )
    parser.add_argument(
        "--shared-prefix-reused",
        choices=("true", "false"),
        help="Optionally keep only runs with this shared-prefix reuse setting",
    )
    args = parser.parse_args()

    benchmarks = [item for item in args.benchmarks.split(",") if item]
    if not benchmarks:
        parser.error("--benchmarks must not be empty")
    if args.min_atomic_subcircuits < 0:
        parser.error("--min-atomic-subcircuits must be nonnegative")
    rows = export_subset(
        args.input_csv,
        args.output_csv,
        qpu_layout=args.qpu_layout,
        benchmarks=benchmarks,
        minimum_atomic_subcircuits=args.min_atomic_subcircuits,
        telegate_cache=(
            args.telegate_cache == "true"
            if args.telegate_cache is not None
            else None
        ),
        shared_prefix_reused=(
            args.shared_prefix_reused == "true"
            if args.shared_prefix_reused is not None
            else None
        ),
    )
    print(f"Wrote {len(rows)} rows to {args.output_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
