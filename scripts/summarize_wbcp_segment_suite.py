#!/usr/bin/env python3
"""Aggregate 12-benchmark WBCP segment sweeps into one report."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import statistics
import tempfile
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "navi-matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = (
    "QV",
    "IQP",
    "Random",
    "QFTAdder",
    "PQC",
    "Permutation",
    "BV",
    "SwapTest",
    "MCMT",
    "VQE",
    "Comparator",
    "CuccaroAdder",
)
DEFAULT_SEGMENTS = (5, 10, 15, 20, 25, 30)
CSV_FIELDS = (
    "benchmark",
    "requested_qubits",
    "actual_qubits",
    "circuit_depth",
    "qpu_layout",
    "network_topology",
    "circuit_seed",
    "best_requested_segments",
    "best_actual_segments",
    "quality_tied_requested_segments",
    "best_epr_pairs",
    "best_remote_effective_fidelity",
    "target_20_actual_segments",
    "target_20_epr_pairs",
    "target_20_epr_gap_to_best",
    "legacy_actual_segments",
    "legacy_epr_pairs",
    "legacy_epr_gap_to_best_candidate",
    "overall_best_configuration",
    "overall_best_actual_segments",
    "overall_best_epr_pairs",
    "source_summary",
)


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _read_summary(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return loaded


def _candidate_sort_key(row: dict[str, Any]) -> tuple[float, float, float]:
    fidelity = row.get("remote_effective_fidelity")
    return (
        float(row["epr_pairs"]),
        -float(fidelity) if fidelity is not None else 0.0,
        float(row["total_time_sec"]),
    )


def collect_suite(
    input_root: Path,
    requested_qubits: int,
    qpu_layout: str,
    circuit_seed: int,
    segments: Sequence[int] = DEFAULT_SEGMENTS,
    network_topology: str = "all_to_all",
) -> tuple[list[dict[str, Any]], dict[str, dict[int, int]]]:
    expected_segments = tuple(int(value) for value in segments)
    matches: dict[str, list[tuple[Path, dict[str, Any]]]] = {
        benchmark: [] for benchmark in BENCHMARKS
    }
    for path in sorted(input_root.glob("*/summary.json")):
        summary = _read_summary(path)
        experiment = summary.get("experiment", {})
        if (
            experiment.get("requested_qubits") != requested_qubits
            or experiment.get("qpu_layout") != qpu_layout
            or experiment.get("circuit_seed") != circuit_seed
            or experiment.get("network_topology") != network_topology
            or experiment.get("window_length_bounds") != [30, 500]
        ):
            continue
        benchmark = experiment.get("benchmark")
        rows = summary.get("rows", [])
        if (
            benchmark in matches
            and rows
            and all(row.get("evaluation_mode") == "communication" for row in rows)
        ):
            matches[benchmark].append((path, summary))

    missing = [benchmark for benchmark, found in matches.items() if not found]
    duplicates = [benchmark for benchmark, found in matches.items() if len(found) > 1]
    if missing:
        raise ValueError(f"Missing benchmark summaries: {', '.join(missing)}")
    if duplicates:
        raise ValueError(f"Duplicate benchmark summaries: {', '.join(duplicates)}")

    suite_rows: list[dict[str, Any]] = []
    epr_matrix: dict[str, dict[int, int]] = {}
    for benchmark in BENCHMARKS:
        path, summary = matches[benchmark][0]
        experiment = summary["experiment"]
        all_rows = summary["rows"]
        legacy = next(
            row for row in all_rows if row.get("requested_segments") is None
        )
        candidates = {
            int(row["requested_segments"]): row
            for row in all_rows
            if row.get("requested_segments") is not None
        }
        absent = [value for value in expected_segments if value not in candidates]
        if absent:
            raise ValueError(
                f"{benchmark} is missing target segment counts: "
                + ", ".join(str(value) for value in absent)
            )
        selected = [candidates[value] for value in expected_segments]
        best_epr = min(int(row["epr_pairs"]) for row in selected)
        quality_ties = [
            row for row in selected if int(row["epr_pairs"]) == best_epr
        ]
        recommended = min(quality_ties, key=_candidate_sort_key)
        target_20 = candidates.get(20)
        overall = min([legacy, *selected], key=_candidate_sort_key)
        epr_matrix[benchmark] = {
            value: int(candidates[value]["epr_pairs"])
            for value in expected_segments
        }
        suite_rows.append(
            {
                "benchmark": benchmark,
                "requested_qubits": experiment["requested_qubits"],
                "actual_qubits": experiment["actual_qubits"],
                "circuit_depth": experiment["circuit_depth"],
                "qpu_layout": experiment["qpu_layout"],
                "network_topology": experiment["network_topology"],
                "circuit_seed": experiment["circuit_seed"],
                "best_requested_segments": recommended["requested_segments"],
                "best_actual_segments": recommended["actual_segments"],
                "quality_tied_requested_segments": ",".join(
                    str(row["requested_segments"]) for row in quality_ties
                ),
                "best_epr_pairs": best_epr,
                "best_remote_effective_fidelity": recommended[
                    "remote_effective_fidelity"
                ],
                "target_20_actual_segments": (
                    target_20["actual_segments"] if target_20 is not None else None
                ),
                "target_20_epr_pairs": (
                    target_20["epr_pairs"] if target_20 is not None else None
                ),
                "target_20_epr_gap_to_best": (
                    (int(target_20["epr_pairs"]) - best_epr) / best_epr
                    if target_20 is not None and best_epr
                    else None
                ),
                "legacy_actual_segments": legacy["actual_segments"],
                "legacy_epr_pairs": legacy["epr_pairs"],
                "legacy_epr_gap_to_best_candidate": (
                    (int(legacy["epr_pairs"]) - best_epr) / best_epr
                    if best_epr
                    else None
                ),
                "overall_best_configuration": overall["configuration"],
                "overall_best_actual_segments": overall["actual_segments"],
                "overall_best_epr_pairs": overall["epr_pairs"],
                "source_summary": str(path.relative_to(input_root)),
            }
        )
    return suite_rows, epr_matrix


def write_suite(
    output_dir: Path,
    rows: Sequence[dict[str, Any]],
    epr_matrix: dict[str, dict[int, int]],
    segments: Sequence[int] = DEFAULT_SEGMENTS,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "suite_summary.csv"
    json_path = output_dir / "suite_summary.json"
    png_path = output_dir / "wbcp_segment_suite.png"
    pdf_path = output_dir / "wbcp_segment_suite.pdf"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    target_gaps = [float(row["target_20_epr_gap_to_best"]) for row in rows]
    best_epr_total = sum(int(row["best_epr_pairs"]) for row in rows)
    target_20_epr_total = sum(int(row["target_20_epr_pairs"]) for row in rows)
    candidate_totals = {
        int(value): sum(epr_matrix[benchmark][int(value)] for benchmark in epr_matrix)
        for value in segments
    }
    candidate_mean_gaps = {
        int(value): statistics.mean(
            (epr_matrix[str(row["benchmark"])][int(value)] - int(row["best_epr_pairs"]))
            / int(row["best_epr_pairs"])
            for row in rows
        )
        for value in segments
    }
    report = {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "requested_qubits": rows[0]["requested_qubits"],
        "qpu_layout": rows[0]["qpu_layout"],
        "network_topology": rows[0]["network_topology"],
        "circuit_seed": rows[0]["circuit_seed"],
        "benchmark_count": len(rows),
        "candidate_segments": list(segments),
        "selection_rule": (
            "minimum EPR pairs; ties use maximum remote effective fidelity, "
            "then minimum total runtime"
        ),
        "quality_ties_are_reported_separately": True,
        "target_20_quality_best_benchmark_count": sum(gap == 0 for gap in target_gaps),
        "best_candidate_epr_pairs_total": best_epr_total,
        "target_20_epr_pairs_total": target_20_epr_total,
        "target_20_aggregate_epr_gap_to_best": (
            target_20_epr_total / best_epr_total - 1.0
            if best_epr_total
            else None
        ),
        "target_20_mean_per_benchmark_epr_gap_to_best": statistics.mean(target_gaps),
        "target_20_median_per_benchmark_epr_gap_to_best": statistics.median(target_gaps),
        "candidate_epr_pairs_totals": candidate_totals,
        "best_fixed_segments_by_total_epr": min(candidate_totals, key=candidate_totals.get),
        "candidate_mean_per_benchmark_epr_gap_to_best": candidate_mean_gaps,
        "best_fixed_segments_by_mean_relative_epr_gap": min(
            candidate_mean_gaps,
            key=candidate_mean_gaps.get,
        ),
        "rows": list(rows),
        "epr_pairs_by_benchmark_and_requested_segments": epr_matrix,
    }
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    segment_values = [int(value) for value in segments]
    benchmark_values = [str(row["benchmark"]) for row in rows]
    raw = np.asarray(
        [
            [epr_matrix[benchmark][value] for value in segment_values]
            for benchmark in benchmark_values
        ],
        dtype=float,
    )
    minima = np.min(raw, axis=1, keepdims=True)
    overhead = np.divide(raw - minima, minima, out=np.zeros_like(raw), where=minima != 0)

    figure, axis = plt.subplots(figsize=(9.2, 6.5))
    image = axis.imshow(overhead * 100.0, cmap="YlOrRd", aspect="auto", vmin=0)
    axis.set_xticks(range(len(segment_values)), [str(value) for value in segment_values])
    axis.set_yticks(range(len(benchmark_values)), benchmark_values)
    axis.set_xlabel("Requested WBCP segments")
    axis.set_ylabel("Benchmark")
    axis.set_title(
        "WBCP EPR overhead relative to each benchmark's best candidate\n"
        f"({rows[0]['qpu_layout']} QPUs, {rows[0]['network_topology']}, "
        f"requested {rows[0]['requested_qubits']} qubits, seed {rows[0]['circuit_seed']})"
    )
    for row_index in range(raw.shape[0]):
        for column_index in range(raw.shape[1]):
            is_best = raw[row_index, column_index] == minima[row_index, 0]
            label = f"{int(raw[row_index, column_index])}"
            if is_best:
                label += " *"
            axis.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=7.2,
                color="white" if overhead[row_index, column_index] > 0.45 else "black",
            )
    colorbar = figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label("EPR overhead vs. row minimum (%)")
    figure.tight_layout()
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return {"csv": csv_path, "json": json_path, "png": png_path, "pdf": pdf_path}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="outputs/wbcp_segment_analysis")
    parser.add_argument("--output-dir")
    parser.add_argument("--requested-qubits", type=int, default=200)
    parser.add_argument("--qpu-layout", default="4x50")
    parser.add_argument("--circuit-seed", type=int, default=42)
    parser.add_argument(
        "--network",
        choices=("all_to_all", "chain", "star", "mesh_grid"),
        default="all_to_all",
    )
    parser.add_argument(
        "--segments",
        default=",".join(str(value) for value in DEFAULT_SEGMENTS),
    )
    args = parser.parse_args(argv)
    try:
        args.segments = tuple(int(value) for value in args.segments.split(","))
    except ValueError:
        parser.error("--segments must be a comma-separated list of integers")
    if not args.segments or any(value <= 0 for value in args.segments):
        parser.error("--segments values must be positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_root = _resolve_path(args.input_root)
    output_dir = (
        _resolve_path(args.output_dir)
        if args.output_dir
        else input_root
        / (
            f"suite_{args.requested_qubits}q_{args.qpu_layout}_"
            f"seed{args.circuit_seed}_{args.network}_bounded_communication"
        )
    )
    rows, epr_matrix = collect_suite(
        input_root,
        args.requested_qubits,
        args.qpu_layout,
        args.circuit_seed,
        args.segments,
        args.network,
    )
    artifacts = write_suite(output_dir, rows, epr_matrix, args.segments)
    print(f"[WBCP_SUITE] benchmarks={len(rows)}")
    for row in rows:
        print(
            f"[WBCP_SUITE] {row['benchmark']}: "
            f"best_target={row['best_requested_segments']}, "
            f"actual={row['best_actual_segments']}, "
            f"EPR={row['best_epr_pairs']}, "
            f"quality_ties={row['quality_tied_requested_segments']}"
        )
    for name, path in artifacts.items():
        print(f"[WBCP_SUITE] {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
