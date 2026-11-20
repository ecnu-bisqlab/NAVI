#!/usr/bin/env python3
"""Build one reusable NAVI zero-cut and partition-assignment context."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.navi import InsufficientAtomicZeroCutSubcircuitsError, NAVI
from src.utils import Backend, Network, get_config, select_circuit


BUNDLE_VERSION = 1


def _csv_ints(value: str) -> list[int]:
    try:
        parsed = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("all values must be positive integers")
    return parsed


def _csv_strings(value: str) -> list[str]:
    parsed = [item.strip() for item in value.split(",")]
    if not parsed or any(not item for item in parsed):
        raise argparse.ArgumentTypeError("expected comma-separated nonempty values")
    return parsed


def _expand(values: list[Any], count: int, name: str) -> list[Any]:
    if len(values) == 1:
        return values * count
    if len(values) != count:
        raise ValueError(f"{name} contains {len(values)} values; expected 1 or {count}")
    return values


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def _atomic_write_pickle(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted((_json_safe(item) for item in value), key=str)
    item_method = getattr(value, "item", None)
    if callable(item_method):
        return _json_safe(item_method())
    return str(value)


def _build_compile_config(global_config: dict[str, Any], circuit_name: str) -> dict[str, Any]:
    compiler_config = global_config.get("compiler_config", {})
    return {
        "circuit_name": circuit_name,
        "_output_folder": str(global_config.get("output_folder", "outputs")),
        **global_config.get("compile_config", {}),
        **compiler_config.get("common", {}),
        **compiler_config.get("navi", {}),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-bundle", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--shared-prefix-id", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--qubits", type=int, required=True)
    parser.add_argument("--cores", type=int, required=True)
    parser.add_argument("--capacity", type=_csv_ints, required=True)
    parser.add_argument("--network", required=True)
    parser.add_argument("--backend", type=_csv_strings, required=True)
    parser.add_argument("--date", type=_csv_strings, required=True)
    parser.add_argument("--circuit-seed", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    capacities = _expand(args.capacity, args.cores, "capacity")
    backends = _expand(args.backend, args.cores, "backend")
    dates = _expand(args.date, args.cores, "date")
    global_config = get_config(str(args.config.resolve()))

    setup_start = time.perf_counter()
    network_config = {**global_config.get("network_config", {}), "type": args.network}
    comm_slot_reserve = Network.resolve_comm_slot_reserve(network_config, global_config)
    backend_list = []
    for capacity, backend_name, calibration_date in zip(capacities, backends, dates):
        backend_config = {
            "backend_name": f"{backend_name}_sampled_{capacity + comm_slot_reserve}q",
            "date": dt.datetime.strptime(calibration_date, "%Y-%m-%d"),
        }
        backend_list.append(Backend(global_config, backend_config))
    network_config.setdefault("fidelity_range", [0.90, 0.93])
    network = Network(network_config, backend_list)

    circuit_start = time.perf_counter()
    _, transpiled_circuit, task_info = select_circuit(
        args.benchmark,
        args.qubits,
        args.cores,
        capacities,
        network.basis_gates,
        network.two_qubit_gates,
        circuit_seed=args.circuit_seed,
    )
    circuit_setup_time = time.perf_counter() - circuit_start
    circuit_name = f"{args.benchmark}{transpiled_circuit.num_qubits}"
    compile_config = _build_compile_config(global_config, circuit_name)
    compile_config["benchmark"] = args.benchmark
    compile_config["circuit_seed"] = args.circuit_seed

    compiler = NAVI()
    try:
        shared_context, shared_prefix_time = compiler.build_shared_prefix_context(
            transpiled_circuit,
            network,
            compile_config,
        )
    except InsufficientAtomicZeroCutSubcircuitsError:
        return 78

    setup_time = time.perf_counter() - setup_start
    metadata = {
        "bundle_version": BUNDLE_VERSION,
        "shared_prefix_id": args.shared_prefix_id,
        "benchmark": args.benchmark,
        "circuit_seed": args.circuit_seed,
        "requested_qubit_count": args.qubits,
        "requested_qpu_count": args.cores,
        "requested_qpu_capacities": capacities,
        "network_topology": args.network,
        "backend": backends,
        "calibration_date": dates,
        "circuit_name": circuit_name,
        "task_info": _json_safe(task_info),
        "K": len(shared_context.subc_ranges),
        "shared_prefix_time_sec": float(shared_prefix_time),
        "circuit_setup_time_sec": float(circuit_setup_time),
        "shared_setup_time_sec": float(setup_time),
        "shared_prefix_signature": list(compiler.get_shared_prefix_signature(compile_config)),
    }
    bundle = {
        "bundle_version": BUNDLE_VERSION,
        "metadata": metadata,
        "shared_context": shared_context,
        "task_info": task_info,
    }
    _atomic_write_pickle(args.output_bundle, bundle)
    metadata["bundle_sha256"] = _sha256(args.output_bundle)
    metadata["bundle_path"] = str(args.output_bundle.resolve())
    _atomic_write_json(args.output_manifest, metadata)
    print("[COMMOPT_SHARED_PREFIX] " + json.dumps(metadata, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[ERROR] Failed to build CommOpt shared prefix: {exc}", file=sys.stderr)
        raise
