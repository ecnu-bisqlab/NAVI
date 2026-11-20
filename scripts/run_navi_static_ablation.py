#!/usr/bin/env python3
"""Run audited static-partition NAVI variants without SliceOpt."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_naive_ablation import (
    atomic_write_rows,
    build_network,
    load_ablation_specs,
    reconstruct_checkpoint_circuit,
    validate_circuit,
)
from scripts.run_navi_leave_one_out import (
    CHECKPOINT_SOURCES,
    circuit_fingerprint,
    config_fingerprint,
)
from scripts.insert_navi_static_ablation import insert_static_results
from src.compiler import CompilerUtils
from src.navi import NAVI


PRIORITY_BENCHMARKS = ("BV", "SwapTest", "Comparator", "IQP", "QFTAdder")
VARIANT_ID = "static_commopt_mapopt"
VARIANT_NAME = "Single-Layer Static+CommOpt+MapOpt"
UNIFORM_K_VARIANT_NAME = "K-Matched Uniform Static+CommOpt+MapOpt"

DETAIL_FIELDS = (
    "Circuit",
    "Variant ID",
    "Variant",
    "#Qubits",
    "#Gates",
    "#2Q Gates",
    "Depth",
    "#QPUs",
    "QPUs",
    "net_type",
    "F_norm",
    "Cumulative Error",
    "EPR Pair Consumption",
    "Execution Time",
    "Segment Count",
    "Static Cross-QPU 2Q Gates",
    "Pre-Map Record Count",
    "Pre-Map TeleData Record Count",
    "Pre-Map TeleGate Record Count",
    "Post-Map Record Count",
    "Post-Map TeleData Record Count",
    "Post-Map TeleGate Record Count",
    "Mapper",
    "TeleGate Total Calls",
    "TeleGate Cache Hits",
    "TeleGate Relaxed Cache Hits",
    "TeleGate Cache Misses",
    "Full Span Safeguard Enabled",
    "CommOpt Full Span Won",
    "Source Checkpoint",
    "Circuit Fingerprint",
    "Config Fingerprint",
    "Raw JSON",
    "Evaluated JSON",
    "MapOpt Selection JSON",
)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def _compiler_config(
    global_config: dict[str, Any],
    circuit_name: str,
    output_base: Path,
    *,
    static_segmentation_mode: str = "single_layer",
    static_segment_budgets: dict[str, int] | None = None,
) -> dict[str, Any]:
    config = copy.deepcopy(global_config.get("compiler_config", {}).get("navi", {}))
    config.pop("ablation_mode", None)
    config.update({
        "hybrid_full_span_safeguard": False,
        "benchmark": circuit_name,
        "circuit_name": circuit_name,
        "evaluator_policy": "full_realistic",
        "record_output_base": str(output_base),
        "_output_folder": str(output_base.parent),
    })
    if static_segmentation_mode != "single_layer":
        config["static_segmentation_mode"] = static_segmentation_mode
    if static_segment_budgets is not None:
        config["static_segment_budgets"] = dict(static_segment_budgets)
    return config


def load_static_segment_budgets() -> OrderedDict[str, int]:
    """Read only the Full SliceOpt segment count K from pinned checkpoints."""
    budgets: OrderedDict[str, int] = OrderedDict()
    for circuit_name, source_path in CHECKPOINT_SOURCES.items():
        source = _resolve(source_path)
        payload = json.loads(source.read_text(encoding="utf-8"))
        records = payload.get("records")
        if not isinstance(records, list) or not records:
            raise ValueError(
                f"Full NAVI-Td checkpoint has no records for {circuit_name}: {source}"
            )
        budgets[circuit_name] = len(records)
    return budgets


def _parse_benchmarks(raw: str) -> list[str]:
    if raw.strip().lower() == "priority":
        return list(PRIORITY_BENCHMARKS)
    if raw.strip().lower() == "all":
        return list(CHECKPOINT_SOURCES)
    selected = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in selected if item not in CHECKPOINT_SOURCES]
    if unknown:
        raise ValueError(f"unknown benchmarks: {unknown}")
    return selected


def _load_completed(path: Path) -> OrderedDict[str, dict[str, Any]]:
    completed: OrderedDict[str, dict[str, Any]] = OrderedDict()
    if not path.exists():
        return completed
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = [field for field in DETAIL_FIELDS if field not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"detail CSV is missing fields {missing}: {path}")
        for row in reader:
            circuit = str(row["Circuit"])
            if circuit in completed:
                raise ValueError(f"duplicate completed result for {circuit}: {path}")
            completed[circuit] = row
    return completed


def _static_cross_qpu_twoq_gates(circuit: Any, partition: list[list[int]]) -> int:
    owner = {
        int(qubit): int(qpu)
        for qpu, group in enumerate(partition)
        for qubit in group
    }
    qindex = {qubit: index for index, qubit in enumerate(circuit.qubits)}
    count = 0
    for instruction in circuit.data:
        if len(instruction.qubits) != 2:
            continue
        left, right = (qindex[qubit] for qubit in instruction.qubits)
        count += owner[left] != owner[right]
    return count


def _raw_is_full_span_telegate(path: Path, circuit_depth: int) -> bool:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = list(payload.get("records", []))
    return bool(
        len(records) == 1
        and records[0].get("mapping_type") in {"cat", "telegate"}
        and int(records[0].get("layer_start", -1)) == 0
        and int(records[0].get("layer_end", -1)) == int(circuit_depth) - 1
    )


def run(args: argparse.Namespace) -> None:
    source_csv = _resolve(args.ablation_csv)
    global_config_path = _resolve(args.global_config)
    detail_csv = _resolve(args.detail_csv)
    record_dir = _resolve(args.record_dir)
    global_config = json.loads(global_config_path.read_text(encoding="utf-8"))
    specs = load_ablation_specs(source_csv)
    selected = _parse_benchmarks(args.benchmarks)
    completed = _load_completed(detail_csv)
    static_segment_budgets = None
    if args.static_segmentation_mode == "uniform_k":
        static_segment_budgets = load_static_segment_budgets()
        print(
            "[STATIC_K_BUDGETS] "
            + json.dumps(static_segment_budgets, sort_keys=True),
            flush=True,
        )

    for circuit_name in selected:
        source = _resolve(CHECKPOINT_SOURCES[circuit_name])
        circuit = reconstruct_checkpoint_circuit(source)
        spec = specs[circuit_name]
        validate_circuit(circuit, spec, source)
        fingerprint = circuit_fingerprint(circuit)
        output_base = record_dir / circuit_name / circuit_name
        compiler_config = _compiler_config(
            global_config,
            circuit_name,
            output_base,
            static_segmentation_mode=args.static_segmentation_mode,
            static_segment_budgets=static_segment_budgets,
        )
        cfg_fingerprint = config_fingerprint(
            global_config,
            compiler_config,
            VARIANT_ID,
        )
        raw_path = Path(str(output_base) + "_raw.json")
        evaluated_path = Path(str(output_base) + "_evaluated.json")
        selection_path = Path(str(output_base) + "_mapopt_selection.json")

        if circuit_name in completed and not args.force:
            row = completed[circuit_name]
            expected = {
                "Source Checkpoint": str(source),
                "Circuit Fingerprint": fingerprint,
                "Config Fingerprint": cfg_fingerprint,
                "Raw JSON": str(raw_path),
                "Evaluated JSON": str(evaluated_path),
                "MapOpt Selection JSON": str(selection_path),
            }
            mismatched = {
                field: (row.get(field), value)
                for field, value in expected.items()
                if row.get(field) != value
            }
            if mismatched:
                raise ValueError(
                    f"completed provenance mismatch for {circuit_name}: {mismatched}"
                )
            if not all(path.is_file() for path in (raw_path, evaluated_path, selection_path)):
                raise FileNotFoundError(f"completed audit files are missing for {circuit_name}")
            inferred_full_span = _raw_is_full_span_telegate(
                raw_path,
                int(spec["Depth"]),
            )
            if str(row.get("CommOpt Full Span Won", "")).lower() != str(inferred_full_span).lower():
                row["CommOpt Full Span Won"] = inferred_full_span
                ordered = [completed[name] for name in CHECKPOINT_SOURCES if name in completed]
                atomic_write_rows(detail_csv, ordered, DETAIL_FIELDS)
            print(f"[STATIC_SKIP] {circuit_name}: verified completed result", flush=True)
            continue

        network = build_network(global_config, spec)
        static_partition = CompilerUtils.allocate_qubits(circuit.num_qubits, network)
        static_cross_gates = _static_cross_qpu_twoq_gates(circuit, static_partition)
        print(
            f"[STATIC_START] {circuit_name}: layers={circuit.depth()}, "
            f"static_cross_2q={static_cross_gates}, partition_sizes="
            f"{[len(group) for group in static_partition]}",
            flush=True,
        )
        result, stats = NAVI().compile_leave_one_out(
            circuit,
            network,
            VARIANT_ID,
            compiler_config,
        )
        costs = result.total_costs
        row = {
            "Circuit": circuit_name,
            "Variant ID": VARIANT_ID,
            "Variant": (
                UNIFORM_K_VARIANT_NAME
                if args.static_segmentation_mode == "uniform_k"
                else VARIANT_NAME
            ),
            "#Qubits": int(spec["#Qubits"]),
            "#Gates": int(spec["#Gates"]),
            "#2Q Gates": int(spec["#2Q Gates"]),
            "Depth": int(spec["Depth"]),
            "#QPUs": int(spec["#QPUs"]),
            "QPUs": json.dumps(spec["QPUs"]),
            "net_type": spec["net_type"],
            "F_norm": math.exp(float(costs.total_fidelity_log_sum) / int(spec["Depth"])),
            "Cumulative Error": float(costs.total_fidelity_loss),
            "EPR Pair Consumption": int(costs.epairs),
            "Execution Time": float(costs.execution_time),
            "Segment Count": int(stats["segment_count"]),
            "Static Cross-QPU 2Q Gates": int(static_cross_gates),
            "Pre-Map Record Count": int(stats["pre_map_record_count"]),
            "Pre-Map TeleData Record Count": int(stats["pre_map_teledata_record_count"]),
            "Pre-Map TeleGate Record Count": int(stats["pre_map_telegate_record_count"]),
            "Post-Map Record Count": int(stats["post_map_record_count"]),
            "Post-Map TeleData Record Count": int(stats["post_map_teledata_record_count"]),
            "Post-Map TeleGate Record Count": int(stats["post_map_telegate_record_count"]),
            "Mapper": stats["mapper_id"],
            "TeleGate Total Calls": int(stats["telegate_total_calls"]),
            "TeleGate Cache Hits": int(stats["telegate_cache_hits"]),
            "TeleGate Relaxed Cache Hits": int(stats["telegate_cache_relaxed_hits"]),
            "TeleGate Cache Misses": int(stats["telegate_cache_misses"]),
            "Full Span Safeguard Enabled": bool(
                stats["full_span_safeguard_enabled"]
            ),
            "CommOpt Full Span Won": bool(stats["commopt_full_span_won"]),
            "Source Checkpoint": str(source),
            "Circuit Fingerprint": fingerprint,
            "Config Fingerprint": cfg_fingerprint,
            "Raw JSON": str(raw_path),
            "Evaluated JSON": str(evaluated_path),
            "MapOpt Selection JSON": str(selection_path),
        }
        if not all(path.is_file() for path in (raw_path, evaluated_path, selection_path)):
            raise FileNotFoundError(f"compiler did not write expected audit files for {circuit_name}")
        completed[circuit_name] = row
        ordered = [completed[name] for name in CHECKPOINT_SOURCES if name in completed]
        atomic_write_rows(detail_csv, ordered, DETAIL_FIELDS)
        print(
            f"[STATIC_DONE] {circuit_name}: F_norm={row['F_norm']:.12g}, "
            f"error={row['Cumulative Error']:.12g}, EPR={row['EPR Pair Consumption']}, "
            f"segments={row['Segment Count']}, records={row['Post-Map Record Count']}",
            flush=True,
        )

    if args.final_csv:
        if set(completed) == set(CHECKPOINT_SOURCES):
            final_csv = _resolve(args.final_csv)
            insert_static_results(final_csv, detail_csv)
            print(
                f"[STATIC_INSERT_DONE] target={final_csv} source={detail_csv}",
                flush=True,
            )
        else:
            missing = sorted(set(CHECKPOINT_SOURCES) - set(completed))
            print(
                f"[STATIC_INSERT_PENDING] missing={missing}",
                flush=True,
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmarks", default="priority")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--ablation-csv", default="outputs/navi_ablation_with_naive.csv")
    parser.add_argument("--global-config", default="config.navi_ablation.json")
    parser.add_argument(
        "--static-segmentation-mode",
        choices=("single_layer", "uniform_k"),
        default="single_layer",
    )
    parser.add_argument(
        "--final-csv",
        default=None,
        help="Atomically insert results after all 12 circuits are complete.",
    )
    parser.add_argument(
        "--detail-csv",
        default=(
            "outputs/navi_static_ablation_legacy_cat_no_fullspan/"
            "variant_results.csv"
        ),
    )
    parser.add_argument(
        "--record-dir",
        default="outputs/navi_static_ablation_legacy_cat_no_fullspan/records",
    )
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
