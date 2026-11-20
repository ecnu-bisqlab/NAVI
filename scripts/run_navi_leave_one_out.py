#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "navi-matplotlib"),
)

from qiskit import QuantumCircuit

from scripts.run_naive_ablation import (
    CHECKPOINT_SOURCES as NAIVE_CHECKPOINT_SOURCES,
    OFFICIAL_METRICS,
    atomic_write_rows,
    build_network,
    load_ablation_specs,
    reconstruct_checkpoint_circuit,
    validate_circuit,
)
from src.navi import NAVI


CHECKPOINT_SOURCES = OrderedDict(NAIVE_CHECKPOINT_SOURCES)
CHECKPOINT_SOURCES["PQC"] = (
    "outputs/PQC80/navi_checkpoints_20260813_091231/NAVI-Td_raw.json"
)

VARIANTS = OrderedDict([
    ("commopt_mapopt", "CommOpt+MapOpt"),
    ("sliceopt_mapopt", "SliceOpt+MapOpt"),
])

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
    "Pre-Map Record Count",
    "Pre-Map TeleData Record Count",
    "Pre-Map TeleGate Record Count",
    "Post-Map Record Count",
    "Post-Map TeleData Record Count",
    "Post-Map TeleGate Record Count",
    "Mapper",
    "Source Checkpoint",
    "Circuit Fingerprint",
    "Config Fingerprint",
    "Raw JSON",
    "Evaluated JSON",
)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def _stable_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _stable_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_stable_value(item) for item in value]
    return str(value)


def _sha256_payload(payload: Any) -> str:
    encoded = json.dumps(
        _stable_value(payload),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def circuit_fingerprint(circuit: QuantumCircuit) -> str:
    qindex = {qubit: index for index, qubit in enumerate(circuit.qubits)}
    cindex = {clbit: index for index, clbit in enumerate(circuit.clbits)}
    operations = []
    for instruction in circuit.data:
        operations.append({
            "name": instruction.operation.name,
            "qubits": [qindex[qubit] for qubit in instruction.qubits],
            "clbits": [cindex[clbit] for clbit in instruction.clbits],
            "params": [str(param) for param in instruction.operation.params],
        })
    return _sha256_payload({
        "num_qubits": circuit.num_qubits,
        "num_clbits": circuit.num_clbits,
        "global_phase": str(circuit.global_phase),
        "operations": operations,
    })


def config_fingerprint(
    global_config: dict[str, Any],
    compiler_config: dict[str, Any],
    variant_id: str,
) -> str:
    excluded = {
        "benchmark",
        "circuit_name",
        "record_output_base",
        "_output_folder",
    }
    stable_compiler_config = {
        key: value for key, value in compiler_config.items() if key not in excluded
    }
    return _sha256_payload({
        "variant": variant_id,
        "compiler_config": stable_compiler_config,
        "network_config": global_config.get("network_config", {}),
        "backend": "ibm_marrakesh",
        "calibration_date": "2026-03-01",
        "qpus": [20, 20, 20, 20],
        "network": "chain",
        "evaluator_policy": "full_realistic",
    })


def parse_benchmarks(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(CHECKPOINT_SOURCES)
    selected = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in selected if item not in CHECKPOINT_SOURCES]
    if unknown:
        raise ValueError(f"unknown benchmarks: {unknown}")
    return selected


def parse_variants(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(VARIANTS)
    aliases = {
        "commopt+mapopt": "commopt_mapopt",
        "sliceopt+mapopt": "sliceopt_mapopt",
    }
    selected = []
    for item in (part.strip() for part in raw.split(",")):
        if not item:
            continue
        normalized = aliases.get(item.lower(), item.lower())
        if normalized not in VARIANTS:
            raise ValueError(f"unknown variant: {item!r}")
        if normalized not in selected:
            selected.append(normalized)
    return selected


def load_completed_rows(path: Path) -> OrderedDict[tuple[str, str], dict[str, Any]]:
    completed: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
    if not path.exists():
        return completed
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing_fields = [field for field in DETAIL_FIELDS if field not in (reader.fieldnames or [])]
        if missing_fields:
            raise ValueError(f"completed detail CSV is missing fields {missing_fields}: {path}")
        for row in reader:
            key = (str(row["Circuit"]), str(row["Variant ID"]))
            if key in completed:
                raise ValueError(f"duplicate completed result {key} in {path}")
            if key[0] not in CHECKPOINT_SOURCES or key[1] not in VARIANTS:
                raise ValueError(f"unknown completed result key {key} in {path}")
            completed[key] = row
    return completed


def ordered_completed_rows(
    completed: dict[tuple[str, str], dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        completed[(circuit, variant)]
        for circuit in CHECKPOINT_SOURCES
        for variant in VARIANTS
        if (circuit, variant) in completed
    ]


def merge_complete_ablation(
    source_csv: Path,
    output_csv: Path,
    completed: dict[tuple[str, str], dict[str, Any]],
) -> None:
    missing = [
        (circuit, variant)
        for circuit in CHECKPOINT_SOURCES
        for variant in VARIANTS
        if (circuit, variant) not in completed
    ]
    if missing:
        raise ValueError(f"cannot build complete ablation CSV; missing results: {missing}")

    with source_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        source_fields = list(reader.fieldnames or [])
        rows = [row for row in reader if row.get("Metrics") in OFFICIAL_METRICS]
    required_methods = {"Naive", "NAVI-Td", "NAVI-Hy", "NAVI"}
    if not required_methods.issubset(source_fields):
        raise ValueError(
            f"source ablation CSV is missing method columns: {sorted(required_methods - set(source_fields))}"
        )

    metadata_fields = source_fields[:source_fields.index("Metrics") + 1]
    output_fields = metadata_fields + [
        "Naive",
        "SliceOpt",
        "CommOpt+MapOpt",
        "SliceOpt+MapOpt",
        "SliceOpt+CommOpt",
        "NAVI",
    ]
    output_rows = []
    for source_row in rows:
        circuit = str(source_row["Circuit"])
        metric = str(source_row["Metrics"])
        output_row = {field: source_row[field] for field in metadata_fields}
        output_row.update({
            "Naive": source_row["Naive"],
            "SliceOpt": source_row["NAVI-Td"],
            "CommOpt+MapOpt": completed[(circuit, "commopt_mapopt")][metric],
            "SliceOpt+MapOpt": completed[(circuit, "sliceopt_mapopt")][metric],
            "SliceOpt+CommOpt": source_row["NAVI-Hy"],
            "NAVI": source_row["NAVI"],
        })
        output_rows.append(output_row)

    expected_rows = len(CHECKPOINT_SOURCES) * len(OFFICIAL_METRICS)
    if len(output_rows) != expected_rows:
        raise ValueError(f"unexpected merged row count: {len(output_rows)} != {expected_rows}")
    atomic_write_rows(output_csv, output_rows, output_fields)


def _compiler_config(
    global_config: dict[str, Any],
    circuit_name: str,
    output_base: Path,
) -> dict[str, Any]:
    config = dict(global_config.get("compiler_config", {}).get("navi", {}))
    config.pop("ablation_mode", None)
    config.update({
        "benchmark": circuit_name,
        "circuit_name": circuit_name,
        "evaluator_policy": "full_realistic",
        "record_output_base": str(output_base),
        "_output_folder": str(output_base.parent),
    })
    return config


def atomic_write_detail(
    path: Path,
    completed: dict[tuple[str, str], dict[str, Any]],
) -> None:
    atomic_write_rows(path, ordered_completed_rows(completed), DETAIL_FIELDS)


def run(args: argparse.Namespace) -> None:
    source_csv = _resolve(args.ablation_csv)
    global_config_path = _resolve(args.global_config)
    detail_csv = _resolve(args.detail_csv)
    merged_csv = _resolve(args.merged_csv)
    record_dir = _resolve(args.record_dir)
    source_hash_before = hashlib.sha256(source_csv.read_bytes()).hexdigest()
    global_config = json.loads(global_config_path.read_text(encoding="utf-8"))
    specs = load_ablation_specs(source_csv)
    selected_benchmarks = parse_benchmarks(args.benchmarks)
    selected_variants = parse_variants(args.variants)

    rebuilt: dict[str, tuple[QuantumCircuit, Path, str]] = {}
    for circuit_name, relative_source in CHECKPOINT_SOURCES.items():
        source = _resolve(relative_source)
        circuit = reconstruct_checkpoint_circuit(source)
        validate_circuit(circuit, specs[circuit_name], source)
        fingerprint = circuit_fingerprint(circuit)
        rebuilt[circuit_name] = (circuit, source, fingerprint)
        print(
            f"[LOO_VALIDATE] {circuit_name}: qubits={circuit.num_qubits}, "
            f"gates={len(circuit.data)}, 2q={sum(len(i.qubits) == 2 for i in circuit.data)}, "
            f"depth={circuit.depth()}, fingerprint={fingerprint}, source={source}"
        )

    if args.validate_only:
        print("[LOO_VALIDATE] all 12 pinned circuits match the source ablation CSV")
        return

    completed = load_completed_rows(detail_csv)
    for circuit_name in selected_benchmarks:
        circuit, source, fingerprint = rebuilt[circuit_name]
        spec = specs[circuit_name]
        for variant_id in selected_variants:
            key = (circuit_name, variant_id)
            output_base = record_dir / variant_id / circuit_name / circuit_name
            compiler_config = _compiler_config(global_config, circuit_name, output_base)
            cfg_fingerprint = config_fingerprint(global_config, compiler_config, variant_id)
            expected_raw = str(output_base) + "_raw.json"
            expected_evaluated = str(output_base) + "_evaluated.json"

            if key in completed and not args.force:
                row = completed[key]
                expected_provenance = {
                    "Source Checkpoint": str(source),
                    "Circuit Fingerprint": fingerprint,
                    "Config Fingerprint": cfg_fingerprint,
                    "Raw JSON": expected_raw,
                    "Evaluated JSON": expected_evaluated,
                }
                mismatched = {
                    field: (row.get(field), value)
                    for field, value in expected_provenance.items()
                    if row.get(field) != value
                }
                if mismatched:
                    raise ValueError(f"completed provenance mismatch for {key}: {mismatched}")
                if not Path(expected_raw).is_file() or not Path(expected_evaluated).is_file():
                    raise FileNotFoundError(f"completed record files are missing for {key}")
                print(f"[LOO_SKIP] {circuit_name}/{variant_id}: verified completed result")
                continue

            network = build_network(global_config, spec)
            result, stats = NAVI().compile_leave_one_out(
                circuit,
                network,
                variant_id,
                compiler_config,
            )
            costs = result.total_costs
            row = {
                "Circuit": circuit_name,
                "Variant ID": variant_id,
                "Variant": VARIANTS[variant_id],
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
                "Pre-Map Record Count": int(stats["pre_map_record_count"]),
                "Pre-Map TeleData Record Count": int(stats["pre_map_teledata_record_count"]),
                "Pre-Map TeleGate Record Count": int(stats["pre_map_telegate_record_count"]),
                "Post-Map Record Count": int(stats["post_map_record_count"]),
                "Post-Map TeleData Record Count": int(stats["post_map_teledata_record_count"]),
                "Post-Map TeleGate Record Count": int(stats["post_map_telegate_record_count"]),
                "Mapper": stats["mapper_id"],
                "Source Checkpoint": str(source),
                "Circuit Fingerprint": fingerprint,
                "Config Fingerprint": cfg_fingerprint,
                "Raw JSON": expected_raw,
                "Evaluated JSON": expected_evaluated,
            }
            if not Path(expected_raw).is_file() or not Path(expected_evaluated).is_file():
                raise FileNotFoundError(f"compiler did not write expected audit files for {key}")
            completed[key] = row
            atomic_write_detail(detail_csv, completed)
            print(
                f"[LOO_DONE] {circuit_name}/{variant_id}: "
                f"F_norm={row['F_norm']:.12g}, error={row['Cumulative Error']:.12g}, "
                f"EPR={row['EPR Pair Consumption']}, segments={row['Segment Count']}, "
                f"records={row['Post-Map Record Count']}"
            )

    source_hash_after = hashlib.sha256(source_csv.read_bytes()).hexdigest()
    if source_hash_after != source_hash_before:
        raise RuntimeError(f"source ablation CSV changed during the run: {source_csv}")

    missing = [
        (circuit, variant)
        for circuit in CHECKPOINT_SOURCES
        for variant in VARIANTS
        if (circuit, variant) not in completed
    ]
    if missing:
        print(f"[LOO_PARTIAL] remaining={missing}")
        return
    merge_complete_ablation(source_csv, merged_csv, completed)
    print(f"[LOO_COMPLETE] detail={detail_csv}")
    print(f"[LOO_COMPLETE] merged={merged_csv}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run NAVI leave-one-out ablations on pinned transpiled circuits."
    )
    parser.add_argument("--benchmarks", default="all", help="all or comma-separated circuit names")
    parser.add_argument(
        "--variants",
        default="all",
        help="all, commopt_mapopt, sliceopt_mapopt, or comma-separated values",
    )
    parser.add_argument("--force", action="store_true", help="rerun selected completed pairs")
    parser.add_argument("--validate-only", action="store_true", help="validate all circuit provenance only")
    parser.add_argument("--ablation-csv", default="outputs/navi_ablation_with_naive.csv")
    parser.add_argument("--global-config", default="config.navi_ablation.json")
    parser.add_argument(
        "--detail-csv",
        default="outputs/navi_leave_one_out/variant_results.csv",
    )
    parser.add_argument("--merged-csv", default="outputs/navi_ablation_complete.csv")
    parser.add_argument("--record-dir", default="outputs/navi_leave_one_out/records")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
