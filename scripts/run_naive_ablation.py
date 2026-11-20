#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import datetime
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

from src.baselines import Naive
from src.navi.navi_compiler import load_mapping_record_list_from_raw_json
from src.utils import Backend, Network


OFFICIAL_METRICS = ("F_norm", "Cumulative Error", "EPR Pair Consumption")
CHECKPOINT_SOURCES = OrderedDict([
    ("BV", "outputs/BV80/navi_checkpoints_20260812_231707/NAVI-Td_raw.json"),
    ("SwapTest", "outputs/SwapTest79/navi_checkpoints_20260812_231715/NAVI-Td_raw.json"),
    ("Permutation", "outputs/Permutation80/navi_checkpoints_20260812_231749/NAVI-Td_raw.json"),
    ("MCMT", "outputs/MCMT80/navi_checkpoints_20260812_231803/NAVI-Td_raw.json"),
    ("Comparator", "outputs/Comparator80/navi_checkpoints_20260812_232514/NAVI-Td_raw.json"),
    ("QFTAdder", "outputs/QFTAdder80/navi_checkpoints_20260812_232600/NAVI-Td_raw.json"),
    ("CuccaroAdder", "outputs/CuccaroAdder80/navi_checkpoints_20260813_000531/NAVI-Td_raw.json"),
    ("PQC", "outputs/PQC80/navi_checkpoints_20260813_000623/NAVI-Td_raw.json"),
    ("IQP", "outputs/IQP80/navi_checkpoints_20260813_001426/NAVI-Td_raw.json"),
    ("Random", "outputs/Random80/navi_checkpoints_20260813_005804/NAVI-Td_raw.json"),
    ("QV", "outputs/QV80/navi_checkpoints_20260813_010934/NAVI-Td_raw.json"),
    ("VQE", "outputs/VQE80/navi_checkpoints_20260813_011738/NAVI-Td_raw.json"),
])

DETAIL_FIELDS = (
    "Circuit",
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
    "TeleGate Count",
    "Execution Time",
    "Partition",
    "Source Checkpoint",
    "Raw JSON",
    "Evaluated JSON",
)


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()


def load_ablation_specs(path: Path) -> OrderedDict[str, dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"ablation CSV is empty: {path}")

    specs: OrderedDict[str, dict[str, Any]] = OrderedDict()
    identity_fields = (
        "#Qubits",
        "#Gates",
        "#2Q Gates",
        "Depth",
        "#QPUs",
        "QPUs",
        "net_type",
    )
    for row in rows:
        circuit = str(row["Circuit"])
        candidate = {field: row[field] for field in identity_fields}
        if circuit in specs:
            if candidate != specs[circuit]:
                raise ValueError(f"inconsistent metadata rows for {circuit}")
        else:
            specs[circuit] = candidate

    expected = list(CHECKPOINT_SOURCES)
    if list(specs) != expected:
        raise ValueError(
            "ablation circuits/order do not match pinned checkpoints: "
            f"csv={list(specs)!r}, checkpoints={expected!r}"
        )
    for circuit, spec in specs.items():
        spec["#Qubits"] = int(spec["#Qubits"])
        spec["#Gates"] = int(spec["#Gates"])
        spec["#2Q Gates"] = int(spec["#2Q Gates"])
        spec["Depth"] = int(spec["Depth"])
        spec["#QPUs"] = int(spec["#QPUs"])
        spec["QPUs"] = [int(value) for value in ast.literal_eval(spec["QPUs"])]
        if spec["#QPUs"] != 4 or spec["QPUs"] != [20, 20, 20, 20]:
            raise ValueError(f"unexpected QPU layout for {circuit}: {spec['QPUs']}")
        if spec["net_type"] != "chain":
            raise ValueError(f"unexpected network topology for {circuit}: {spec['net_type']}")
    return specs


def reconstruct_checkpoint_circuit(path: Path) -> QuantumCircuit:
    record_list = load_mapping_record_list_from_raw_json(path)
    records = record_list.records
    if not records:
        raise ValueError(f"checkpoint has no records: {path}")

    first_start = int(records[0].layer_start)
    if first_start != 0:
        raise ValueError(f"checkpoint does not start at layer 0: {path}")
    for previous, current in zip(records, records[1:]):
        if int(current.layer_start) != int(previous.layer_end) + 1:
            raise ValueError(f"checkpoint layer ranges are not contiguous: {path}")

    circuits: list[QuantumCircuit] = []
    for index, record in enumerate(records):
        extra_info = record.extra_info or {}
        ops = extra_info.get("ops")
        if not isinstance(ops, QuantumCircuit):
            raise TypeError(f"checkpoint record {index} has no serialized circuit ops: {path}")
        circuits.append(ops)

    num_qubits = circuits[0].num_qubits
    if any(circuit.num_qubits != num_qubits for circuit in circuits):
        raise ValueError(f"checkpoint record qubit counts differ: {path}")
    rebuilt = QuantumCircuit(num_qubits)
    for circuit in circuits:
        rebuilt.compose(circuit, inplace=True)
    return rebuilt


def validate_circuit(circuit: QuantumCircuit, spec: dict[str, Any], source: Path) -> None:
    observed = {
        "#Qubits": int(circuit.num_qubits),
        "#Gates": int(len(circuit.data)),
        "#2Q Gates": int(sum(1 for instruction in circuit.data if len(instruction.qubits) == 2)),
        "Depth": int(circuit.depth()),
    }
    expected = {key: int(spec[key]) for key in observed}
    if observed != expected:
        raise ValueError(
            f"checkpoint circuit does not match ablation metadata: source={source}, "
            f"expected={expected}, observed={observed}"
        )


def build_network(global_config: dict[str, Any], spec: dict[str, Any]) -> Network:
    network_config = {
        **global_config.get("network_config", {}),
        "type": str(spec["net_type"]),
    }
    reserve = Network.resolve_comm_slot_reserve(network_config, global_config)
    calibration_date = datetime.datetime.strptime("2026-03-01", "%Y-%m-%d")
    backends = []
    for capacity in spec["QPUs"]:
        sampled_capacity = int(capacity) + reserve
        backends.append(Backend(global_config, {
            "backend_name": f"ibm_marrakesh_sampled_{sampled_capacity}q",
            "date": calibration_date,
        }))
    return Network(network_config, backends)


def atomic_write_rows(path: Path, rows: Iterable[dict[str, Any]], fieldnames: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_completed_rows(path: Path) -> OrderedDict[str, dict[str, Any]]:
    completed: OrderedDict[str, dict[str, Any]] = OrderedDict()
    if not path.exists():
        return completed
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            circuit = str(row.get("Circuit", ""))
            if not circuit or circuit in completed:
                raise ValueError(f"invalid or duplicate completed row in {path}: {circuit!r}")
            missing = [field for field in DETAIL_FIELDS if field not in row]
            if missing:
                raise ValueError(f"completed result row is missing fields {missing}: {path}")
            completed[circuit] = row
    return completed


def merge_ablation_results(
    source_csv: Path,
    output_csv: Path,
    completed: dict[str, dict[str, Any]],
) -> None:
    missing = [circuit for circuit in CHECKPOINT_SOURCES if circuit not in completed]
    if missing:
        raise ValueError(f"cannot build merged ablation CSV; missing Naive results: {missing}")

    with source_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = [row for row in reader if row.get("Metrics") in OFFICIAL_METRICS]
    if "Naive" in fieldnames:
        raise ValueError(f"source ablation CSV already contains a Naive column: {source_csv}")
    fieldnames.append("Naive")
    for row in rows:
        row["Naive"] = completed[row["Circuit"]][row["Metrics"]]
    expected_rows = len(CHECKPOINT_SOURCES) * len(OFFICIAL_METRICS)
    if len(rows) != expected_rows:
        raise ValueError(f"unexpected official metric row count: {len(rows)} != {expected_rows}")
    atomic_write_rows(output_csv, rows, fieldnames)


def parse_benchmarks(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(CHECKPOINT_SOURCES)
    selected = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in selected if item not in CHECKPOINT_SOURCES]
    if unknown:
        raise ValueError(f"unknown benchmarks: {unknown}")
    return selected


def run(args: argparse.Namespace) -> None:
    ablation_csv = _resolve(args.ablation_csv)
    global_config_path = _resolve(args.global_config)
    detail_csv = _resolve(args.detail_csv)
    merged_csv = _resolve(args.merged_csv)
    record_dir = _resolve(args.record_dir)
    global_config = json.loads(global_config_path.read_text(encoding="utf-8"))
    specs = load_ablation_specs(ablation_csv)
    selected = parse_benchmarks(args.benchmarks)

    rebuilt: dict[str, tuple[QuantumCircuit, Path]] = {}
    for circuit_name, relative_source in CHECKPOINT_SOURCES.items():
        source = _resolve(relative_source)
        circuit = reconstruct_checkpoint_circuit(source)
        validate_circuit(circuit, specs[circuit_name], source)
        rebuilt[circuit_name] = (circuit, source)
        print(
            f"[NAIVE_VALIDATE] {circuit_name}: qubits={circuit.num_qubits}, "
            f"gates={len(circuit.data)}, depth={circuit.depth()}, source={source}"
        )

    if args.validate_only:
        print("[NAIVE_VALIDATE] all 12 pinned checkpoint circuits match the ablation CSV")
        return

    completed = load_completed_rows(detail_csv)
    for circuit_name in selected:
        circuit, source = rebuilt[circuit_name]
        expected_source = str(source)
        if circuit_name in completed and not args.force:
            if completed[circuit_name]["Source Checkpoint"] != expected_source:
                raise ValueError(f"completed result provenance mismatch for {circuit_name}")
            print(f"[NAIVE_SKIP] {circuit_name}: completed result already exists")
            continue

        spec = specs[circuit_name]
        network = build_network(global_config, spec)
        output_base = record_dir / circuit_name / circuit_name
        compiler_config = {
            "circuit_name": circuit_name,
            "record_output_base": str(output_base),
            "evaluator_policy": "full_realistic",
        }
        result = Naive().compile(circuit, network, compiler_config)
        costs = result.total_costs
        telegate_count = int(costs.telegate_exec_events)
        epairs = int(costs.epairs)
        if telegate_count != int(costs.cat_ents):
            raise RuntimeError(
                f"Naive CAT/TeleGate count mismatch for {circuit_name}: "
                f"cat_ents={costs.cat_ents}, telegate_exec_events={telegate_count}"
            )
        if epairs < telegate_count:
            raise RuntimeError(
                f"Naive EPR count is smaller than TeleGate count for {circuit_name}: "
                f"epairs={epairs}, telegates={telegate_count}"
            )

        partition = result.records[0].partition
        expected_partition = []
        next_qubit = 0
        for capacity in spec["QPUs"]:
            end = min(next_qubit + int(capacity), circuit.num_qubits)
            expected_partition.append(list(range(next_qubit, end)))
            next_qubit = end
        if partition != expected_partition:
            raise RuntimeError(f"unexpected Naive partition for {circuit_name}: {partition}")

        row = {
            "Circuit": circuit_name,
            "#Qubits": int(spec["#Qubits"]),
            "#Gates": int(spec["#Gates"]),
            "#2Q Gates": int(spec["#2Q Gates"]),
            "Depth": int(spec["Depth"]),
            "#QPUs": int(spec["#QPUs"]),
            "QPUs": json.dumps(spec["QPUs"]),
            "net_type": spec["net_type"],
            "F_norm": math.exp(float(costs.total_fidelity_log_sum) / int(spec["Depth"])),
            "Cumulative Error": float(costs.total_fidelity_loss),
            "EPR Pair Consumption": epairs,
            "TeleGate Count": telegate_count,
            "Execution Time": float(costs.execution_time),
            "Partition": json.dumps(partition, separators=(",", ":")),
            "Source Checkpoint": expected_source,
            "Raw JSON": str(output_base) + "_raw.json",
            "Evaluated JSON": str(output_base) + "_evaluated.json",
        }
        completed[circuit_name] = row
        ordered_rows = [completed[name] for name in CHECKPOINT_SOURCES if name in completed]
        atomic_write_rows(detail_csv, ordered_rows, DETAIL_FIELDS)
        print(
            f"[NAIVE_DONE] {circuit_name}: F_norm={row['F_norm']:.12g}, "
            f"error={row['Cumulative Error']:.12g}, EPR={epairs}, "
            f"TeleGates={telegate_count}"
        )

    if all(circuit in completed for circuit in CHECKPOINT_SOURCES):
        merge_ablation_results(ablation_csv, merged_csv, completed)
        print(f"[NAIVE_COMPLETE] detail={detail_csv}")
        print(f"[NAIVE_COMPLETE] merged={merged_csv}")
    else:
        remaining = [circuit for circuit in CHECKPOINT_SOURCES if circuit not in completed]
        print(f"[NAIVE_PARTIAL] remaining={remaining}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the pinned NAVI ablation circuits with contiguous static TeleGate placement."
    )
    parser.add_argument("--benchmarks", default="all", help="all or a comma-separated benchmark list")
    parser.add_argument("--force", action="store_true", help="rerun selected completed benchmarks")
    parser.add_argument("--validate-only", action="store_true", help="validate checkpoint provenance without evaluation")
    parser.add_argument("--ablation-csv", default="outputs/navi_ablation.csv")
    parser.add_argument("--global-config", default="config.navi_ablation.json")
    parser.add_argument("--detail-csv", default="outputs/navi_naive_ablation/naive_results.csv")
    parser.add_argument("--merged-csv", default="outputs/navi_ablation_with_naive.csv")
    parser.add_argument("--record-dir", default="outputs/navi_naive_ablation/records")
    return parser


if __name__ == "__main__":
    run(build_parser().parse_args())
