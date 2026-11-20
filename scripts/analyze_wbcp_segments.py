#!/usr/bin/env python3
"""Sweep WBCP segment counts and report quality/runtime trade-offs."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import platform
import sys
import tempfile
import time
from contextlib import ExitStack, nullcontext
from pathlib import Path
from typing import Any, Sequence
from unittest.mock import patch

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "navi-matplotlib"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import qiskit


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines import WBCP
from src.compiler import CompilerUtils
from src.compiler import evaluator as evaluator_module
from src.utils import Backend, Network, select_circuit
from src.utils.qcircuits import SUPPORTED_BENCHMARKS


DEFAULT_SEGMENTS = (5, 10, 15, 20, 25, 30)
CSV_FIELDS = (
    "benchmark",
    "circuit_seed",
    "requested_qubits",
    "actual_qubits",
    "qpu_layout",
    "evaluation_mode",
    "configuration",
    "reused_from_configuration",
    "requested_segments",
    "window_length",
    "actual_segments",
    "partition_changes",
    "epr_pairs",
    "effective_fidelity",
    "remote_effective_fidelity",
    "cumulative_error",
    "remote_communications",
    "pre_evaluation_time_sec",
    "evaluation_time_sec",
    "total_time_sec",
)


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def _read_config(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    config = dict(loaded)
    device_folder = Path(str(config["device_properties_folder"])).expanduser()
    if not device_folder.is_absolute():
        device_folder = (REPO_ROOT / device_folder).resolve()
    config["device_properties_folder"] = f"{device_folder}{os.sep}"
    return config


def _partition_signature(partition: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(sorted(group)) for group in partition)


def _partition_changes(records: Sequence[Any]) -> int:
    signatures = [_partition_signature(record.partition) for record in records]
    return sum(left != right for left, right in zip(signatures, signatures[1:]))


def _build_inputs(args: argparse.Namespace) -> tuple[
    dict[str, Any], dict[str, Any], list[Backend], Any, dict[str, Any]
]:
    config_path = _resolve_path(args.config)
    config = _read_config(config_path)
    calibration_date = dt.datetime.strptime(args.calibration_date, "%Y-%m-%d")
    network_config = {
        **config.get("network_config", {}),
        "type": args.network,
        "seed": args.network_seed,
    }
    reserve = Network.resolve_comm_slot_reserve(network_config, config)
    sampled_capacity = args.capacity + reserve
    backend_config = {
        "backend_name": f"{args.backend}_sampled_{sampled_capacity}q",
        "date": calibration_date,
    }
    backends = [Backend(config, backend_config) for _ in range(args.cores)]
    preparation_network = Network(network_config, backends)
    _, circuit, task_info = select_circuit(
        args.benchmark,
        args.qubits,
        args.cores,
        [args.capacity] * args.cores,
        preparation_network.basis_gates,
        preparation_network.two_qubit_gates,
        circuit_seed=args.circuit_seed,
    )
    metadata = {
        "benchmark": args.benchmark,
        "circuit_seed": args.circuit_seed,
        "network_seed": args.network_seed,
        "requested_qubits": args.qubits,
        "actual_qubits": circuit.num_qubits,
        "circuit_depth": circuit.depth(),
        "gate_count": task_info["#Gates"],
        "two_qubit_gate_count": task_info["#2Q Gates"],
        "qpu_layout": f"{args.cores}x{args.capacity}",
        "network_topology": args.network,
        "backend": args.backend,
        "sampled_backend_capacity": sampled_capacity,
        "calibration_date": args.calibration_date,
        "oee_iteration_cap": args.oee_iterations,
        "window_length_bounds": [30, 500],
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }
    return config, network_config, backends, circuit, metadata


def run_sweep(
    args: argparse.Namespace,
    config: dict[str, Any],
    network_config: dict[str, Any],
    backends: list[Backend],
    circuit: Any,
    progress_path: Path | None = None,
) -> list[dict[str, Any]]:
    wbcp = WBCP()
    circuit_layers = CompilerUtils.build_circuit_layers(circuit)
    configurations: list[tuple[str, int | None]] = [("legacy_default", None)]
    configurations.extend((f"target_{count}", count) for count in args.segments)
    rows: list[dict[str, Any]] = []
    if progress_path is not None and progress_path.exists():
        loaded = json.loads(progress_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise ValueError(f"Expected a JSON array: {progress_path}")
        rows = loaded
        for row in rows:
            row.setdefault("evaluation_mode", "full")
            row.setdefault("remote_effective_fidelity", None)
            row.setdefault("reused_from_configuration", None)
            if row["evaluation_mode"] != args.evaluation_mode:
                raise ValueError(
                    f"Progress file evaluation mode is {row['evaluation_mode']!r}, "
                    f"not {args.evaluation_mode!r}: {progress_path}"
                )
    completed_labels = {str(row["configuration"]) for row in rows}

    for label, requested_segments in configurations:
        if label in completed_labels:
            print(f"[WBCP_SEGMENTS] SKIP completed {label}", flush=True)
            continue
        run_config = {
            **config.get("compile_config", {}),
            **config.get("compiler_config", {}).get("common", {}),
            **config.get("compiler_config", {}).get("wbcp", {}),
            "iteration": args.oee_iterations,
            "benchmark": args.benchmark,
            "circuit_seed": args.circuit_seed,
            "circuit_name": f"{args.benchmark}{circuit.num_qubits}",
        }
        if requested_segments is not None:
            run_config["num_segments"] = requested_segments

        window_length, actual_segments = WBCP._resolve_window_plan(
            circuit.depth(),
            run_config,
        )
        if not 30 <= window_length <= 500:
            raise AssertionError(
                f"WBCP window length outside [30, 500]: {window_length}"
            )
        print(
            f"[WBCP_SEGMENTS] START {label}: requested={requested_segments}, "
            f"window_length={window_length}, actual={actual_segments}",
            flush=True,
        )
        equivalent = next(
            (
                row
                for row in rows
                if int(row["window_length"]) == window_length
                and int(row["actual_segments"]) == actual_segments
            ),
            None,
        )
        if equivalent is not None:
            row = dict(equivalent)
            row.update(
                {
                    "configuration": label,
                    "reused_from_configuration": equivalent["configuration"],
                    "requested_segments": requested_segments,
                    "window_length": window_length,
                    "actual_segments": actual_segments,
                }
            )
            rows.append(row)
            if progress_path is not None:
                progress_path.parent.mkdir(parents=True, exist_ok=True)
                progress_path.write_text(
                    json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
            print(
                f"[WBCP_SEGMENTS] REUSE {label} from "
                f"{equivalent['configuration']}: epairs={row['epr_pairs']}",
                flush=True,
            )
            continue
        network = Network(network_config, backends)
        started = time.perf_counter()
        raw_records = wbcp._k_way_WBCP(
            circuit,
            network,
            args.oee_iterations,
            ilp_output=False,
            config=run_config,
        )
        raw_records.ensure_record_ops(circuit, circuit_layers)
        pre_evaluation_time = time.perf_counter() - started
        if args.evaluation_mode == "communication":
            def communication_transpile(local_circuit: Any, **kwargs: Any) -> Any:
                del kwargs
                # EPR usage is determined while replaying remote operations.
                # Returning the unrouted local buffer keeps the evaluator's
                # identity wire layout while avoiding irrelevant local routing.
                return local_circuit

            def skip_local_costs(*args: Any, **kwargs: Any) -> int:
                del args, kwargs
                return 0

            def skip_local_ops(*args: Any, **kwargs: Any) -> None:
                del args, kwargs

            evaluation_context = ExitStack()
            evaluation_context.enter_context(
                patch.object(
                    evaluator_module,
                    "transpile",
                    side_effect=communication_transpile,
                )
            )
            evaluation_context.enter_context(
                patch.object(
                    evaluator_module.MappingEvaluator,
                    "_accumulate_local_circuit_costs",
                    side_effect=skip_local_costs,
                )
            )
            evaluation_context.enter_context(
                patch.object(
                    evaluator_module.MappingEvaluator,
                    "add_local_ops",
                    side_effect=skip_local_ops,
                )
            )
        else:
            evaluation_context = nullcontext()
        with evaluation_context:
            evaluated, _, evaluation_time = wbcp._evaluate_raw_mapping_records_with_timing(
                raw_records,
                network,
                policy_name=run_config.get("evaluator_policy"),
                config=run_config,
                pre_eval_time=pre_evaluation_time,
            )
        total_time = time.perf_counter() - started
        costs = evaluated.total_costs
        effective_fidelity = (
            math.exp(float(costs.total_fidelity_log_sum) / float(circuit.depth()))
            if args.evaluation_mode == "full"
            else None
        )
        remote_effective_fidelity = math.exp(
            float(costs.remote_fidelity_log_sum) / float(circuit.depth())
        )
        row = {
            "benchmark": args.benchmark,
            "circuit_seed": args.circuit_seed,
            "requested_qubits": args.qubits,
            "actual_qubits": circuit.num_qubits,
            "qpu_layout": f"{args.cores}x{args.capacity}",
            "evaluation_mode": args.evaluation_mode,
            "configuration": label,
            "reused_from_configuration": None,
            "requested_segments": requested_segments,
            "window_length": window_length,
            "actual_segments": actual_segments,
            "partition_changes": _partition_changes(evaluated.records),
            "epr_pairs": int(costs.epairs),
            "effective_fidelity": effective_fidelity,
            "remote_effective_fidelity": remote_effective_fidelity,
            "cumulative_error": float(
                costs.total_fidelity_loss
                if args.evaluation_mode == "full"
                else costs.remote_fidelity_loss
            ),
            "remote_communications": int(costs.num_comms),
            "pre_evaluation_time_sec": pre_evaluation_time,
            "evaluation_time_sec": evaluation_time,
            "total_time_sec": total_time,
        }
        rows.append(row)
        if progress_path is not None:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            progress_path.write_text(
                json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        print(
            f"[WBCP_SEGMENTS] DONE {label}: epairs={row['epr_pairs']}, "
            f"remote_F_eff={remote_effective_fidelity:.6f}, "
            f"total_time={total_time:.3f}s",
            flush=True,
        )
    return rows


def _write_outputs(
    output_dir: Path,
    rows: Sequence[dict[str, Any]],
    metadata: dict[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "segment_sweep.csv"
    json_path = output_dir / "summary.json"
    png_path = output_dir / "wbcp_segment_sensitivity.png"
    pdf_path = output_dir / "wbcp_segment_sensitivity.pdf"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    explicit_rows = [row for row in rows if row["requested_segments"] is not None]
    target_20 = next((row for row in explicit_rows if row["requested_segments"] == 20), None)
    best_epr_row = min(explicit_rows, key=lambda row: row["epr_pairs"])
    fidelity_key = (
        "effective_fidelity"
        if all(row.get("effective_fidelity") is not None for row in explicit_rows)
        else "remote_effective_fidelity"
    )
    best_fidelity_row = max(explicit_rows, key=lambda row: row[fidelity_key])
    best_epr = best_epr_row["epr_pairs"]
    best_fidelity = best_fidelity_row[fidelity_key]
    legacy = rows[0]
    summary = {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "experiment": metadata,
        "legacy_default": legacy,
        "target_20": target_20,
        "best_observed_epr_pairs": best_epr,
        "best_observed_epr_configuration": best_epr_row["configuration"],
        "best_observed_effective_fidelity": best_fidelity,
        "best_observed_fidelity_scope": fidelity_key,
        "best_observed_fidelity_configuration": best_fidelity_row["configuration"],
        "target_20_epr_gap_to_best": (
            (target_20["epr_pairs"] - best_epr) / best_epr
            if target_20 is not None and best_epr
            else None
        ),
        "target_20_vs_legacy": (
            {
                "relative_epr_change": (
                    target_20["epr_pairs"] - legacy["epr_pairs"]
                )
                / legacy["epr_pairs"],
                "relative_effective_fidelity_change": (
                    target_20[fidelity_key] - legacy[fidelity_key]
                )
                / legacy[fidelity_key],
                "relative_total_time_change": (
                    target_20["total_time_sec"] - legacy["total_time_sec"]
                )
                / legacy["total_time_sec"],
            }
            if target_20 is not None
            else None
        ),
        "all_window_lengths_within_bounds": all(
            30 <= int(row["window_length"]) <= 500 for row in rows
        ),
        "rows": list(rows),
        "software": {
            "python": platform.python_version(),
            "qiskit": qiskit.__version__,
            "networkx": nx.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    labels = [
        "default\n" + str(row["actual_segments"])
        if row["requested_segments"] is None
        else str(row["requested_segments"]) + "\n" + str(row["actual_segments"])
        for row in rows
    ]
    x_values = list(range(len(rows)))
    figure, (quality_axis, time_axis) = plt.subplots(1, 2, figsize=(11.2, 4.5))
    epr_values = [row["epr_pairs"] for row in rows]
    fidelity_values = [row.get(fidelity_key) for row in rows]
    epr_line = quality_axis.plot(
        x_values, epr_values, color="#2369a1", marker="o", label="EPR pairs"
    )
    fidelity_axis = quality_axis.twinx()
    fidelity_line = fidelity_axis.plot(
        x_values,
        fidelity_values,
        color="#238b57",
        marker="s",
        label=(
            "Effective fidelity"
            if fidelity_key == "effective_fidelity"
            else "Remote effective fidelity"
        ),
    )
    quality_axis.set_xticks(x_values, labels)
    quality_axis.set_xlabel("Requested segments / actual windows")
    quality_axis.set_ylabel("EPR pairs", color="#2369a1")
    fidelity_axis.set_ylabel(
        "Effective fidelity"
        if fidelity_key == "effective_fidelity"
        else "Remote effective fidelity",
        color="#238b57",
    )
    quality_axis.grid(True, alpha=0.25)
    lines = epr_line + fidelity_line
    quality_axis.legend(lines, [line.get_label() for line in lines], loc="best")

    total_times = [row["total_time_sec"] for row in rows]
    time_axis.bar(x_values, total_times, color="#7464a7")
    time_axis.set_xticks(x_values, labels)
    time_axis.set_xlabel("Requested segments / actual windows")
    time_axis.set_ylabel("End-to-end WBCP time (s)")
    time_axis.grid(axis="y", alpha=0.25)
    figure.suptitle(
        f"WBCP segment sensitivity on {metadata['benchmark']}-{metadata['actual_qubits']} "
        f"({metadata['qpu_layout']} QPUs, seed {metadata['circuit_seed']})"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return {"csv": csv_path, "json": json_path, "png": png_path, "pdf": pdf_path}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.example.json")
    parser.add_argument("--output-dir")
    parser.add_argument("--benchmark", choices=SUPPORTED_BENCHMARKS, default="Random")
    parser.add_argument("--qubits", type=int, default=200)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--capacity", type=int, default=50)
    parser.add_argument("--network", default="all_to_all")
    parser.add_argument("--backend", default="ibm_marrakesh")
    parser.add_argument("--calibration-date", default="2026-03-01")
    parser.add_argument("--circuit-seed", type=int, default=42)
    parser.add_argument("--network-seed", type=int, default=42)
    parser.add_argument("--oee-iterations", type=int, default=50)
    parser.add_argument(
        "--evaluation-mode",
        choices=("full", "communication"),
        default="full",
    )
    parser.add_argument(
        "--segments",
        default=",".join(str(value) for value in DEFAULT_SEGMENTS),
        help="Comma-separated target segment counts.",
    )
    args = parser.parse_args(argv)
    try:
        args.segments = tuple(int(value) for value in args.segments.split(","))
    except ValueError:
        parser.error("--segments must be a comma-separated list of integers")
    if not args.segments or any(value <= 0 for value in args.segments):
        parser.error("--segments values must be positive")
    if len(set(args.segments)) != len(args.segments):
        parser.error("--segments values must be unique")
    if args.qubits > args.cores * args.capacity:
        parser.error("--qubits must not exceed --cores multiplied by --capacity")
    if args.oee_iterations <= 0:
        parser.error("--oee-iterations must be positive")
    try:
        dt.datetime.strptime(args.calibration_date, "%Y-%m-%d")
    except ValueError:
        parser.error("--calibration-date must use YYYY-MM-DD")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config, network_config, backends, circuit, metadata = _build_inputs(args)
    output_dir = (
        _resolve_path(args.output_dir)
        if args.output_dir
        else REPO_ROOT
        / "outputs"
        / "wbcp_segment_analysis"
        / (
            f"{args.benchmark.lower()}_{args.qubits}q_"
            f"{args.cores}x{args.capacity}_seed{args.circuit_seed}_"
            f"{args.network}_bounded"
            + ("" if args.evaluation_mode == "full" else "_communication")
        )
    )
    progress_path = output_dir / "progress.json"
    rows = run_sweep(
        args,
        config,
        network_config,
        backends,
        circuit,
        progress_path=progress_path,
    )
    artifacts = _write_outputs(output_dir, rows, metadata)
    for name, path in artifacts.items():
        print(f"[WBCP_SEGMENTS] {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
