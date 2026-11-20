#!/usr/bin/env python3
"""Aggregate per-benchmark OEE iteration analyses into a suite report."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
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
CSV_FIELDS = (
    "benchmark",
    "requested_qubits",
    "actual_qubits",
    "qpu_layout",
    "circuit_seed",
    "initial_weighted_cut",
    "accepted_iterations",
    "first_unchanged_cap",
    "target_cap",
    "target_weighted_cut",
    "reference_cap",
    "reference_weighted_cut",
    "cut_reduction",
    "relative_cut_reduction",
    "target_reference_match",
    "sufficient",
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


def collect_suite(
    input_root: Path,
    requested_qubits: int,
    qpu_layout: str,
    circuit_seed: int,
) -> list[dict[str, Any]]:
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
        ):
            continue
        benchmark = experiment.get("benchmark")
        if benchmark in matches:
            matches[benchmark].append((path, summary))

    missing = [benchmark for benchmark, found in matches.items() if not found]
    duplicates = [benchmark for benchmark, found in matches.items() if len(found) > 1]
    if missing:
        raise ValueError(f"Missing benchmark summaries: {', '.join(missing)}")
    if duplicates:
        raise ValueError(f"Duplicate benchmark summaries: {', '.join(duplicates)}")

    rows: list[dict[str, Any]] = []
    for benchmark in BENCHMARKS:
        path, summary = matches[benchmark][0]
        experiment = summary["experiment"]
        convergence = summary["convergence"]
        initial_cut = float(convergence["initial_weighted_cut"])
        target_cut = float(convergence["target_weighted_cut"])
        reduction = initial_cut - target_cut
        rows.append(
            {
                "benchmark": benchmark,
                "requested_qubits": experiment["requested_qubits"],
                "actual_qubits": experiment["actual_qubits"],
                "qpu_layout": experiment["qpu_layout"],
                "circuit_seed": experiment["circuit_seed"],
                "initial_weighted_cut": convergence["initial_weighted_cut"],
                "accepted_iterations": convergence["accepted_iterations"],
                "first_unchanged_cap": convergence["first_unchanged_cap"],
                "target_cap": convergence["target_cap"],
                "target_weighted_cut": convergence["target_weighted_cut"],
                "reference_cap": convergence["reference_cap"],
                "reference_weighted_cut": convergence["reference_weighted_cut"],
                "cut_reduction": int(reduction) if reduction.is_integer() else reduction,
                "relative_cut_reduction": reduction / initial_cut if initial_cut else 0.0,
                "target_reference_match": convergence["target_reference_match"],
                "sufficient": convergence["sufficient"],
                "source_summary": str(path.relative_to(input_root)),
            }
        )
    return rows


def write_suite(
    output_dir: Path,
    rows: Sequence[dict[str, Any]],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "suite_summary.csv"
    json_path = output_dir / "suite_summary.json"
    png_path = output_dir / "s_oee_suite_iterations.png"
    pdf_path = output_dir / "s_oee_suite_iterations.pdf"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    target_caps = {int(row["target_cap"]) for row in rows}
    reference_caps = {int(row["reference_cap"]) for row in rows}
    if len(target_caps) != 1 or len(reference_caps) != 1:
        raise ValueError("Suite summaries must use common target and reference caps.")
    target_cap = next(iter(target_caps))
    reference_cap = next(iter(reference_caps))
    all_sufficient = all(bool(row["sufficient"]) for row in rows)
    max_iterations = max(int(row["accepted_iterations"]) for row in rows)
    report = {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "requested_qubits": rows[0]["requested_qubits"],
        "qpu_layout": rows[0]["qpu_layout"],
        "circuit_seed": rows[0]["circuit_seed"],
        "benchmark_count": len(rows),
        "target_cap": target_cap,
        "reference_cap": reference_cap,
        "maximum_accepted_iterations": max_iterations,
        "all_target_reference_match": all(
            bool(row["target_reference_match"]) for row in rows
        ),
        "all_sufficient": all_sufficient,
        "rows": list(rows),
    }
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    labels = [str(row["benchmark"]) for row in rows]
    iterations = [int(row["accepted_iterations"]) for row in rows]
    colors = ["#238b57" if row["sufficient"] else "#c74440" for row in rows]
    figure, (cap_axis, detail_axis) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(10.8, 5.6),
        gridspec_kw={"height_ratios": (1, 3), "hspace": 0.06},
    )
    bars = detail_axis.bar(labels, iterations, color=colors, width=0.72)
    cap_axis.axhline(
        target_cap,
        color="#c74440",
        linestyle="--",
        linewidth=1.8,
        label=f"Configured cap = {target_cap}",
    )
    detail_axis.bar_label(bars, padding=3, fontsize=8)
    detail_axis.set_ylim(0, max_iterations + 2)
    cap_axis.set_ylim(target_cap - 2, target_cap + 2)
    cap_axis.spines.bottom.set_visible(False)
    detail_axis.spines.top.set_visible(False)
    cap_axis.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    cap_axis.tick_params(axis="y", labelsize=8)
    detail_axis.tick_params(axis="x", rotation=35)
    detail_axis.grid(axis="y", alpha=0.25)
    cap_axis.grid(axis="y", alpha=0.25)
    cap_axis.legend(loc="upper right")

    break_size = 0.012
    cap_kwargs = {"transform": cap_axis.transAxes, "color": "black", "clip_on": False}
    cap_axis.plot((-break_size, +break_size), (-break_size, +break_size), **cap_kwargs)
    cap_axis.plot((1 - break_size, 1 + break_size), (-break_size, +break_size), **cap_kwargs)
    detail_kwargs = {
        "transform": detail_axis.transAxes,
        "color": "black",
        "clip_on": False,
    }
    detail_axis.plot((-break_size, +break_size), (1 - break_size, 1 + break_size), **detail_kwargs)
    detail_axis.plot(
        (1 - break_size, 1 + break_size),
        (1 - break_size, 1 + break_size),
        **detail_kwargs,
    )

    figure.supylabel("Effective OEE iterations before convergence")
    figure.suptitle(
        "S-OEE convergence across 12 benchmarks "
        f"({rows[0]['qpu_layout']} QPUs, requested {rows[0]['requested_qubits']} qubits)",
        y=0.99,
    )
    figure.subplots_adjust(left=0.09, right=0.98, top=0.91, bottom=0.22)
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return {"csv": csv_path, "json": json_path, "png": png_path, "pdf": pdf_path}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", default="outputs/oee_iteration_analysis")
    parser.add_argument("--output-dir")
    parser.add_argument("--requested-qubits", type=int, default=200)
    parser.add_argument("--qpu-layout", default="4x50")
    parser.add_argument("--circuit-seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_root = _resolve_path(args.input_root)
    output_dir = (
        _resolve_path(args.output_dir)
        if args.output_dir
        else input_root
        / f"suite_{args.requested_qubits}q_{args.qpu_layout}_seed{args.circuit_seed}"
    )
    rows = collect_suite(
        input_root,
        args.requested_qubits,
        args.qpu_layout,
        args.circuit_seed,
    )
    artifacts = write_suite(output_dir, rows)
    print(
        "[OEE_SUITE] "
        f"benchmarks={len(rows)}, "
        f"maximum_accepted_iterations={max(row['accepted_iterations'] for row in rows)}, "
        f"all_sufficient={str(all(row['sufficient'] for row in rows)).lower()}"
    )
    for name, path in artifacts.items():
        print(f"[OEE_SUITE] {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
