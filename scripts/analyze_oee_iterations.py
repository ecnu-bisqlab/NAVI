#!/usr/bin/env python3
"""Measure whether the Static OEE iteration cap is sufficient for one circuit."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Sequence

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

from src.compiler import CompilerUtils
from src.partitioning import OEE
from src.utils import Backend, Network, select_circuit
from src.utils.qcircuits import SUPPORTED_BENCHMARKS


Partition = list[list[int]]
PartitionSignature = tuple[tuple[int, ...], ...]
Partitioner = Callable[[Partition, nx.Graph, Any, int], Partition]

CSV_FIELDS = (
    "benchmark",
    "circuit_seed",
    "requested_qubits",
    "actual_qubits",
    "qpu_layout",
    "network_topology",
    "iteration_cap",
    "sample_kind",
    "weighted_cut",
    "cut_reduction",
    "relative_cut_reduction",
    "partition_runtime_sec",
    "partition_sha256",
    "same_as_prior_sampled_cap",
    "matches_converged_partition",
)


def canonicalize_partition(partition: Sequence[Sequence[int]]) -> PartitionSignature:
    """Return a stable representation while preserving QPU labels."""
    return tuple(tuple(sorted(int(qubit) for qubit in subset)) for subset in partition)


def partition_sha256(signature: PartitionSignature) -> str:
    payload = json.dumps(signature, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def weighted_cut(qig: nx.Graph, partition: Sequence[Sequence[int]]) -> int | float:
    """Sum interaction weights whose endpoints belong to different QPUs."""
    owner: dict[int, int] = {}
    for part_id, subset in enumerate(partition):
        for qubit in subset:
            if qubit in owner:
                raise ValueError(f"Qubit {qubit} occurs in more than one partition.")
            owner[int(qubit)] = part_id

    missing = set(qig.nodes()) - set(owner)
    if missing:
        raise ValueError(f"Partition is missing QIG nodes: {sorted(missing)}")

    value = float(
        sum(
            float(data.get("weight", 1))
            for left, right, data in qig.edges(data=True)
            if owner[left] != owner[right]
        )
    )
    return int(value) if value.is_integer() else value


def run_cap_sweep(
    initial_partition: Partition,
    qig: nx.Graph,
    network: Any,
    target_cap: int,
    reference_cap: int,
    partitioner: Partitioner = OEE.partition,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run consecutive caps to convergence, then verify target/reference caps."""
    if target_cap < 1:
        raise ValueError("target_cap must be positive.")
    if reference_cap < target_cap:
        raise ValueError("reference_cap must be greater than or equal to target_cap.")

    initial_cut = weighted_cut(qig, initial_partition)
    rows_by_cap: dict[int, dict[str, Any]] = {}
    signatures: dict[int, PartitionSignature] = {}
    sample_tags: dict[int, set[str]] = {}

    def run_cap(cap: int, tag: str) -> dict[str, Any]:
        sample_tags.setdefault(cap, set()).add(tag)
        if cap in rows_by_cap:
            return rows_by_cap[cap]

        started = time.perf_counter()
        partition = partitioner(initial_partition, qig, network, cap)
        runtime = time.perf_counter() - started
        signature = canonicalize_partition(partition)
        cut = weighted_cut(qig, partition)
        reduction = float(initial_cut) - float(cut)
        relative_reduction = reduction / float(initial_cut) if initial_cut else 0.0
        row = {
            "iteration_cap": cap,
            "weighted_cut": cut,
            "cut_reduction": int(reduction) if reduction.is_integer() else reduction,
            "relative_cut_reduction": relative_reduction,
            "partition_runtime_sec": runtime,
            "partition_sha256": partition_sha256(signature),
            "same_as_prior_sampled_cap": None,
            "matches_converged_partition": None,
        }
        rows_by_cap[cap] = row
        signatures[cap] = signature
        return row

    first_unchanged_cap: int | None = None
    previous_signature: PartitionSignature | None = None
    for cap in range(reference_cap + 1):
        row = run_cap(cap, "convergence_scan")
        current_signature = signatures[cap]
        if previous_signature is not None:
            row["same_as_prior_sampled_cap"] = current_signature == previous_signature
            if current_signature == previous_signature:
                first_unchanged_cap = cap
                break
        previous_signature = current_signature

    target_row = run_cap(target_cap, "target_cap")
    reference_row = run_cap(reference_cap, "reference_cap")

    converged_signature = (
        signatures[first_unchanged_cap]
        if first_unchanged_cap is not None
        else None
    )
    for cap, row in rows_by_cap.items():
        if converged_signature is not None:
            row["matches_converged_partition"] = signatures[cap] == converged_signature
        row["sample_kind"] = "+".join(sorted(sample_tags[cap]))

    target_reference_match = (
        signatures[target_cap] == signatures[reference_cap]
        and target_row["weighted_cut"] == reference_row["weighted_cut"]
    )
    accepted_iterations = (
        first_unchanged_cap - 1 if first_unchanged_cap is not None else None
    )
    sufficient = bool(
        first_unchanged_cap is not None
        and first_unchanged_cap <= target_cap
        and target_reference_match
    )
    convergence = {
        "initial_weighted_cut": initial_cut,
        "first_unchanged_cap": first_unchanged_cap,
        "accepted_iterations": accepted_iterations,
        "termination_check_pass": first_unchanged_cap,
        "target_cap": target_cap,
        "target_weighted_cut": target_row["weighted_cut"],
        "target_partition_sha256": target_row["partition_sha256"],
        "reference_cap": reference_cap,
        "reference_weighted_cut": reference_row["weighted_cut"],
        "reference_partition_sha256": reference_row["partition_sha256"],
        "target_reference_match": target_reference_match,
        "sufficient": sufficient,
    }
    return [rows_by_cap[cap] for cap in sorted(rows_by_cap)], convergence


def _resolve_path(path_text: str, *, base: Path = REPO_ROOT) -> Path:
    path = Path(path_text).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _read_config(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    config = dict(loaded)
    device_folder = Path(str(config["device_properties_folder"])).expanduser()
    if not device_folder.is_absolute():
        device_folder = (REPO_ROOT / device_folder).resolve()
    config["device_properties_folder"] = f"{device_folder}{os.sep}"
    return config


def _build_experiment(args: argparse.Namespace) -> tuple[nx.Graph, Partition, dict[str, Any]]:
    config_path = _resolve_path(args.config)
    config = _read_config(config_path)
    calibration_date = dt.datetime.strptime(args.calibration_date, "%Y-%m-%d")
    network_config = {
        **config.get("network_config", {}),
        "type": args.network,
        "seed": args.network_seed,
    }
    comm_slot_reserve = Network.resolve_comm_slot_reserve(network_config, config)
    sampled_capacity = args.capacity + comm_slot_reserve
    sampled_backend_name = f"{args.backend}_sampled_{sampled_capacity}q"
    backend_config = {
        "backend_name": sampled_backend_name,
        "date": calibration_date,
    }

    preparation_started = time.perf_counter()
    backends = [Backend(config, backend_config) for _ in range(args.cores)]
    network = Network(network_config, backends)
    _, circuit, task_info = select_circuit(
        args.benchmark,
        args.qubits,
        args.cores,
        [args.capacity] * args.cores,
        network.basis_gates,
        network.two_qubit_gates,
        circuit_seed=args.circuit_seed,
    )
    qig = CompilerUtils.build_qubit_interaction_graph(circuit)
    initial_partition = CompilerUtils.allocate_qubits(circuit.num_qubits, network)
    preparation_runtime = time.perf_counter() - preparation_started

    context = {
        "benchmark": args.benchmark,
        "circuit_seed": args.circuit_seed,
        "network_seed": args.network_seed,
        "requested_qubits": args.qubits,
        "actual_qubits": circuit.num_qubits,
        "circuit_depth": circuit.depth(),
        "gate_count": task_info["#Gates"],
        "two_qubit_gate_count": task_info["#2Q Gates"],
        "qig_edge_count": qig.number_of_edges(),
        "qig_total_weight": sum(
            int(data.get("weight", 1)) for _, _, data in qig.edges(data=True)
        ),
        "qpu_count": args.cores,
        "qpu_capacity": args.capacity,
        "qpu_layout": f"{args.cores}x{args.capacity}",
        "comm_slot_reserve": comm_slot_reserve,
        "sampled_backend_capacity": sampled_capacity,
        "network_topology": args.network,
        "backend": args.backend,
        "sampled_backend": sampled_backend_name,
        "calibration_date": args.calibration_date,
        "config_path": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "preparation_runtime_sec": preparation_runtime,
    }
    return qig, initial_partition, {"network": network, "context": context}


def _write_trace_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    context: dict[str, Any],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "benchmark": context["benchmark"],
                    "circuit_seed": context["circuit_seed"],
                    "requested_qubits": context["requested_qubits"],
                    "actual_qubits": context["actual_qubits"],
                    "qpu_layout": context["qpu_layout"],
                    "network_topology": context["network_topology"],
                    **row,
                }
            )


def _plot_convergence(
    output_dir: Path,
    rows: Sequence[dict[str, Any]],
    convergence: dict[str, Any],
    context: dict[str, Any],
) -> tuple[Path, Path]:
    caps = [int(row["iteration_cap"]) for row in rows]
    cuts = [float(row["weighted_cut"]) for row in rows]
    first_unchanged = convergence["first_unchanged_cap"]
    detail_limit = first_unchanged if first_unchanged is not None else min(caps[-1], 10)
    detail_rows = [row for row in rows if row["iteration_cap"] <= detail_limit]
    detail_caps = [int(row["iteration_cap"]) for row in detail_rows]
    detail_cuts = [float(row["weighted_cut"]) for row in detail_rows]

    figure, (detail_axis, overview_axis) = plt.subplots(
        1,
        2,
        figsize=(10.8, 4.4),
        gridspec_kw={"width_ratios": (1.0, 1.35)},
    )
    detail_axis.plot(
        detail_caps,
        detail_cuts,
        color="#2369a1",
        marker="o",
        linewidth=2,
        markersize=5,
    )
    overview_axis.plot(
        caps,
        cuts,
        color="#2369a1",
        marker="o",
        linewidth=2,
        markersize=5,
    )
    overview_axis.axvline(
        convergence["target_cap"],
        color="#c74440",
        linestyle="--",
        linewidth=1.6,
        label=f"Configured cap = {convergence['target_cap']}",
    )

    accepted = convergence["accepted_iterations"]
    if accepted is not None:
        accepted_row = next(row for row in rows if row["iteration_cap"] == accepted)
        accepted_cut = float(accepted_row["weighted_cut"])
        detail_axis.scatter([accepted], [accepted_cut], color="#18864b", s=55, zorder=4)
        overview_axis.scatter([accepted], [accepted_cut], color="#18864b", s=45, zorder=4)
        detail_axis.annotate(
            f"Converged after {accepted} effective iterations",
            xy=(accepted, accepted_cut),
            xytext=(-155, 24),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "color": "#18864b"},
            fontsize=9,
        )

    detail_axis.set_title("Early-iteration detail")
    detail_axis.set_xlabel("Maximum OEE iterations")
    detail_axis.set_ylabel("Weighted cut (cross-QPU two-qubit gates)")
    detail_axis.set_xticks(detail_caps)
    detail_axis.grid(True, alpha=0.25)

    overview_axis.set_title("Configured-cap verification")
    overview_axis.set_xlabel("Maximum OEE iterations")
    overview_axis.grid(True, alpha=0.25)
    overview_axis.legend(loc="best")
    figure.suptitle(
        "S-OEE convergence on "
        f"{context['benchmark']}-{context['actual_qubits']} "
        f"({context['qpu_layout']} QPUs, seed {context['circuit_seed']})",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))

    png_path = output_dir / "s_oee_convergence.png"
    pdf_path = output_dir / "s_oee_convergence.pdf"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return png_path, pdf_path


def _write_outputs(
    output_dir: Path,
    rows: list[dict[str, Any]],
    convergence: dict[str, Any],
    context: dict[str, Any],
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "iteration_trace.csv"
    summary_path = output_dir / "summary.json"
    _write_trace_csv(csv_path, rows, context)
    png_path, pdf_path = _plot_convergence(output_dir, rows, convergence, context)

    summary = {
        "schema_version": 1,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "experiment": context,
        "convergence": convergence,
        "verdict": {
            "sufficient": convergence["sufficient"],
            "claim_scope": (
                "This verdict applies only to the fixed Random circuit instance "
                "and QPU layout recorded in this file."
            ),
            "criterion": (
                "The first unchanged partition cap is no greater than the configured "
                "target, and target/reference partitions and weighted cuts match."
            ),
        },
        "software": {
            "python": platform.python_version(),
            "qiskit": qiskit.__version__,
            "networkx": nx.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "artifacts": {
            "trace_csv": csv_path.name,
            "figure_png": png_path.name,
            "figure_pdf": pdf_path.name,
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "csv": csv_path,
        "summary": summary_path,
        "png": png_path,
        "pdf": pdf_path,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.example.json")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--benchmark",
        choices=SUPPORTED_BENCHMARKS,
        default="Random",
    )
    parser.add_argument("--qubits", type=int, default=200)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--capacity", type=int, default=50)
    parser.add_argument("--network", default="all_to_all")
    parser.add_argument("--backend", default="ibm_marrakesh")
    parser.add_argument("--calibration-date", default="2026-03-01")
    parser.add_argument("--circuit-seed", type=int, default=42)
    parser.add_argument("--network-seed", type=int, default=42)
    parser.add_argument("--target-cap", type=int, default=50)
    parser.add_argument("--reference-cap", type=int, default=100)
    args = parser.parse_args(argv)
    for name in ("qubits", "cores", "capacity", "target_cap", "reference_cap"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.reference_cap < args.target_cap:
        parser.error("--reference-cap must be greater than or equal to --target-cap")
    if args.qubits > args.cores * args.capacity:
        parser.error("--qubits must not exceed --cores multiplied by --capacity")
    try:
        dt.datetime.strptime(args.calibration_date, "%Y-%m-%d")
    except ValueError:
        parser.error("--calibration-date must use YYYY-MM-DD")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output_dir:
        output_dir = _resolve_path(args.output_dir)
    else:
        benchmark_slug = args.benchmark.lower().replace("_", "-")
        output_dir = (
            REPO_ROOT
            / "outputs"
            / "oee_iteration_analysis"
            / (
                f"{benchmark_slug}_{args.qubits}q_"
                f"{args.cores}x{args.capacity}_seed{args.circuit_seed}"
            )
        )
    qig, initial_partition, experiment = _build_experiment(args)
    network = experiment.pop("network")
    context = experiment["context"]
    rows, convergence = run_cap_sweep(
        initial_partition,
        qig,
        network,
        args.target_cap,
        args.reference_cap,
    )
    artifacts = _write_outputs(output_dir, rows, convergence, context)

    print(
        "[OEE_ITERATION] "
        f"accepted_iterations={convergence['accepted_iterations']}, "
        f"first_unchanged_cap={convergence['first_unchanged_cap']}, "
        f"cut@{args.target_cap}={convergence['target_weighted_cut']}, "
        f"cut@{args.reference_cap}={convergence['reference_weighted_cut']}, "
        f"sufficient={str(convergence['sufficient']).lower()}"
    )
    for name, path in artifacts.items():
        print(f"[OEE_ITERATION] {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
