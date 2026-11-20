from dataclasses import dataclass, field
from typing import Any, Optional
import time
import sys
import copy
import itertools
import json
import math
import numpy as np
import networkx as nx
import datetime
import os
from pathlib import Path

from qiskit import QuantumCircuit, transpile
from qiskit.converters import circuit_to_dag

from ..compiler import Compiler, CompilerUtils, ExecCosts, MappingRecord, MappingRecordList
from ..compiler.compiler_utils import CommOp
from ..utils import Network, log
from .partitioner import Partitioner, PartitionerFactory
from .partition_assigner import PartitionAssigner, PartitionAssignerFactory
from .telegate_partitioner import TelegatePartitioner, TelegatePartitionerFactory
from .mapper import Mapper, MapperFactory


class InsufficientAtomicZeroCutSubcircuitsError(RuntimeError):
    """Raised when a parameter-analysis circuit has too few atomic blocks."""

    def __init__(self, benchmark: str, actual: int, minimum: int):
        self.benchmark = benchmark
        self.actual = actual
        self.minimum = minimum
        super().__init__(
            f"CommOpt parameter analysis requires at least {minimum} atomic "
            f"zero-cut subcircuits, but {benchmark} produced {actual}."
        )


def _mapopt_float_tolerance(left: float, right: float, atol: float, rtol: float) -> float:
    return float(atol) + float(rtol) * max(abs(float(left)), abs(float(right)))


def select_mapopt_source(
        baseline: ExecCosts,
        candidate: ExecCosts,
        *,
        atol: float = 1e-12,
        rtol: float = 1e-12,
    ) -> tuple[str, str]:
    """Select a MapOpt result, rejecting only dominated or equivalent candidates."""
    baseline_log = float(baseline.total_fidelity_log_sum)
    candidate_log = float(candidate.total_fidelity_log_sum)
    log_tol = _mapopt_float_tolerance(baseline_log, candidate_log, atol, rtol)
    baseline_loss = float(baseline.total_fidelity_loss)
    candidate_loss = float(candidate.total_fidelity_loss)
    loss_tol = _mapopt_float_tolerance(baseline_loss, candidate_loss, atol, rtol)
    baseline_epairs = int(baseline.epairs)
    candidate_epairs = int(candidate.epairs)

    log_equal = abs(baseline_log - candidate_log) <= log_tol
    loss_equal = abs(baseline_loss - candidate_loss) <= loss_tol
    epairs_equal = baseline_epairs == candidate_epairs
    if log_equal and loss_equal and epairs_equal:
        return "no_op", "equivalent_metrics"

    baseline_weakly_better = (
        baseline_log >= candidate_log - log_tol
        and baseline_loss <= candidate_loss + loss_tol
        and baseline_epairs <= candidate_epairs
    )
    baseline_strictly_better = (
        baseline_log > candidate_log + log_tol
        or baseline_loss < candidate_loss - loss_tol
        or baseline_epairs < candidate_epairs
    )
    if baseline_weakly_better and baseline_strictly_better:
        return "no_op", "baseline_dominates_candidate"
    return "candidate", "candidate_not_dominated"


def _mapopt_metric_snapshot(costs: ExecCosts, circuit_depth: int) -> dict[str, Any]:
    log_sum = float(costs.total_fidelity_log_sum)
    return {
        "F_norm": math.exp(log_sum / circuit_depth) if circuit_depth > 0 else 1.0,
        "Cumulative Error": float(costs.total_fidelity_loss),
        "EPR Pair Consumption": int(costs.epairs),
        "total_fidelity_log_sum": log_sum,
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=4, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@dataclass
class CompilationContext:
    """
    存储编译过程中所有中间状态的数据容器
    """
    # --- 输入 (Input) ---
    circuit: QuantumCircuit
    network: Network
    config: dict[str, Any]

    # --- 预处理状态 (Preprocessing State) ---
    dag: Any = None
    circuit_layers: list[list[Any]] = field(default_factory=list) # Any <- DAGOpNode
    multiq_layers: list[list[Any]] = field(default_factory=list)
    map_to_circuit_layer: dict[int, int] = field(default_factory=dict)
    gate_density: float = 0.0
    twoq_gate_density: float = 0.0
    min_depth: int = 0

    # --- 分区表 (Partition Table - P) ---
    P_table: list[list[list[Any]]] = field(default_factory=list)

    # --- 切片表 (Slicing Tables - S, T) ---
    S_table: list[list[int]] = field(default_factory=list)
    T_table: list[list[int]] = field(default_factory=list)
    subc_ranges: list[tuple[int, int]] = field(default_factory=list)

    # --- 结果与计划 (Results & Plans) ---
    partition_candidates: list[Any] = field(default_factory=list)
    partition_plan: list[Any] = field(default_factory=list)
    hybrid_records: MappingRecordList = field(default_factory=MappingRecordList)
    final_records: MappingRecordList = field(default_factory=MappingRecordList)

    # --- 统计信息 (Statistics) ---
    swap_prefix_sums: list[int] = field(default_factory=list)
    epair_prefix_sums: list[int] = field(default_factory=list)
    telegate_cache_hits: int = 0
    telegate_cache_relaxed_hits: int = 0
    telegate_cache_misses: int = 0
    telegate_cache: dict[Any, MappingRecordList] = field(default_factory=dict)
    telegate_total_calls: int = 0
    telegate_strict_seen_keys: set[Any] = field(default_factory=set)
    telegate_range_seen_keys: set[Any] = field(default_factory=set)
    commopt_full_span_won: bool = False
    slice_strategy: str = "greedy_table"
    slice_planning_time_sec: float = 0.0

    # --- 运行时组件 (Runtime Components) ---
    partitioner: Optional[Partitioner] = None
    partition_assigner: Optional[PartitionAssigner] = None
    telegate_partitioner: Optional[TelegatePartitioner] = None
    mapper: Optional[Mapper] = None

    # --- 编译阶段完成标志 ---
    preprocessed: bool = False
    partition_planned: bool = False
    telegate_optimized: bool = False

@dataclass
class HybridSearchState:
    total_fidelity_loss: float
    epairs: float
    records: list[MappingRecord] = field(default_factory=list)
    logical_phy_map: dict[int, tuple[int, int | None]] = field(default_factory=dict)

class NAVI(Compiler):
    """
    Noise-Aware Distributed Quantum Compiler
    """
    compiler_id = "navi"
    SHARED_PREFIX_CONFIG_KEYS = (
        "partitioner",
        "max_option",
        "partition_assigner",
        "slice_strategy",
    )
    SLICE_STRATEGIES = ("dp", "greedy_table")

    def __init__(self):
        super().__init__()
    
    @property
    def name(self) -> str:
        return "NAVI"

    def compile(self, circuit: QuantumCircuit, 
                network: Network, 
                config: Optional[dict[str, Any]] = None) -> MappingRecordList:
        return self._compile_impl(circuit, network, config)

    def get_shared_prefix_signature(self, config: Optional[dict[str, Any]] = None) -> tuple[Any, ...]:
        cfg = config or {}
        legacy_values = tuple(
            cfg.get(key) for key in self.SHARED_PREFIX_CONFIG_KEYS[:-1]
        )
        return legacy_values + (self._resolve_slice_strategy(cfg),)

    @classmethod
    def _resolve_slice_strategy(cls, config: dict[str, Any]) -> str:
        strategy = str(config.get("slice_strategy", "greedy_table")).strip().lower()
        if strategy not in cls.SLICE_STRATEGIES:
            available = ", ".join(cls.SLICE_STRATEGIES)
            raise ValueError(
                f"Unknown slice_strategy: {strategy!r}; available: {available}"
            )
        return strategy

    def build_shared_prefix_context(
        self,
        circuit: QuantumCircuit,
        network: Network,
        config: Optional[dict[str, Any]] = None,
    ) -> tuple[CompilationContext, float]:
        ctx, zero_cut_time = self._build_zero_cut_context(circuit, network, config)
        assign_time = self._assign_partition_plan(ctx)
        return ctx, zero_cut_time + assign_time

    def _build_layerwise_zero_cut_context(
        self,
        circuit: QuantumCircuit,
        network: Network,
        config: Optional[dict[str, Any]] = None,
    ) -> tuple[CompilationContext, float]:
        """Build fixed one-two-qubit-layer segments without SliceOpt."""
        cfg = copy.deepcopy(config or {})
        partitioner_type = cfg.get("partitioner", "recursive_dp")
        max_option = cfg.get("max_option", 1)
        partitioner = PartitionerFactory.create_partitioner(
            partitioner_type,
            network,
            max_options=max_option,
        )
        ctx = CompilationContext(
            circuit=circuit,
            network=network,
            config=cfg,
            partitioner=partitioner,
        )

        start_time = time.time()
        self._step_remove_single_qubit_gates(ctx)
        ctx.subc_ranges = [(index, index) for index in range(len(ctx.multiq_layers))]
        ctx.partition_candidates = []
        capacities = [int(value) for value in network.get_backend_qubit_counts()]

        for index in range(len(ctx.multiq_layers)):
            qig = self._build_qubit_interaction_graph_by_level(ctx, (index, index))
            candidates = self._get_qig_partitions(qig, partitioner)
            if not candidates:
                raise RuntimeError(
                    f"fixed layer segment {index} has no zero-cut partition candidate"
                )
            for partition in candidates:
                if len(partition) != len(capacities):
                    raise RuntimeError(
                        f"fixed layer segment {index} produced {len(partition)} partitions; "
                        f"expected {len(capacities)}"
                    )
                flattened = [qubit for group in partition for qubit in group]
                if sorted(flattened) != list(range(circuit.num_qubits)):
                    raise RuntimeError(
                        f"fixed layer segment {index} partition does not cover each qubit exactly once"
                    )
                if any(len(group) > capacity for group, capacity in zip(partition, capacities)):
                    raise RuntimeError(
                        f"fixed layer segment {index} partition exceeds a QPU capacity"
                    )
            ctx.partition_candidates.append(candidates)

        return ctx, time.time() - start_time

    def _build_layerwise_static_context(
        self,
        circuit: QuantumCircuit,
        network: Network,
        config: Optional[dict[str, Any]] = None,
    ) -> tuple[CompilationContext, float]:
        """Build fixed-partition blocks without invoking SliceOpt.

        This experimental context deliberately avoids the SliceOpt QIG,
        connected-component partitioning, P table, and S/T table.  The same
        contiguous capacity-based partition is reused for every block.  The
        default keeps one block per native multi-qubit layer.  ``uniform_k``
        instead builds K balanced contiguous blocks, where K is supplied as an
        experimental budget rather than inferred from circuit interactions.
        """
        cfg = copy.deepcopy(config or {})
        ctx = CompilationContext(
            circuit=circuit,
            network=network,
            config=cfg,
        )

        start_time = time.time()
        self._step_remove_single_qubit_gates(ctx)
        num_layers = len(ctx.multiq_layers)
        segmentation_mode = str(
            cfg.get("static_segmentation_mode", "single_layer")
        ).strip().lower()
        if segmentation_mode == "single_layer":
            ctx.subc_ranges = [
                (index, index) for index in range(num_layers)
            ]
        elif segmentation_mode == "uniform_k":
            benchmark = str(cfg.get("benchmark", ""))
            budget_map = cfg.get("static_segment_budgets")
            if not isinstance(budget_map, dict) or benchmark not in budget_map:
                raise ValueError(
                    "uniform_k static segmentation requires "
                    "static_segment_budgets[benchmark]"
                )
            budget_raw = budget_map[benchmark]
            if isinstance(budget_raw, bool):
                raise ValueError("static segment budget must be a positive integer")
            try:
                requested_budget = int(budget_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "static segment budget must be a positive integer"
                ) from exc
            if requested_budget <= 0:
                raise ValueError("static segment budget must be a positive integer")

            effective_budget = min(requested_budget, num_layers)
            ctx.subc_ranges = []
            if effective_budget > 0:
                base_width, remainder = divmod(num_layers, effective_budget)
                left = 0
                for block_index in range(effective_budget):
                    width = base_width + int(block_index < remainder)
                    right = left + width - 1
                    ctx.subc_ranges.append((left, right))
                    left = right + 1
                assert left == num_layers
            log(
                "[static_partition] segmentation=uniform_k, "
                f"benchmark={benchmark}, requested_k={requested_budget}, "
                f"effective_k={effective_budget}, layers={num_layers}"
            )
        else:
            raise ValueError(
                "unknown static_segmentation_mode "
                f"{segmentation_mode!r}; expected single_layer or uniform_k"
            )
        static_partition = CompilerUtils.allocate_qubits(circuit.num_qubits, network)
        ctx.partition_candidates = [
            [copy.deepcopy(static_partition)]
            for _ in ctx.subc_ranges
        ]
        return ctx, time.time() - start_time

    @staticmethod
    def _resolve_mapopt_mapper_id(
        config: dict[str, Any],
        records: MappingRecordList,
    ) -> str:
        mapper_id = str(config.get("multi_record_mapper", "boundeddp_neighbor")).lower()
        if mapper_id == "boundeddp_neighbor" and len(records.records) == 1:
            return "single_record_greedy"
        return mapper_id

    def compile_leave_one_out(
        self,
        circuit: QuantumCircuit,
        network: Network,
        variant: str,
        config: Optional[dict[str, Any]] = None,
    ) -> tuple[MappingRecordList, dict[str, Any]]:
        """Compile one leave-one-out NAVI variant and return audit statistics."""
        cfg = copy.deepcopy(config or {})
        normalized_variant = str(variant).strip().lower()
        supported = {
            "commopt_mapopt",
            "sliceopt_mapopt",
            "static_commopt_mapopt",
        }
        if normalized_variant not in supported:
            raise ValueError(
                f"unknown NAVI leave-one-out variant {variant!r}; "
                f"expected one of {sorted(supported)}"
            )

        print(f"Compiling with [NAVI:{normalized_variant}]...")
        if normalized_variant == "commopt_mapopt":
            shared_ctx, prefix_time = self._build_layerwise_zero_cut_context(
                circuit,
                network,
                cfg,
            )
            prefix_time += self._assign_partition_plan(shared_ctx)
            ctx = self._prepare_ctx_from_shared_prefix(shared_ctx, cfg)
            records_start = time.time()
            pre_map_records = self._step_construct_hybrid_records_beam(ctx)
            records_time = time.time() - records_start
        elif normalized_variant == "static_commopt_mapopt":
            # The w/o SliceOpt ablation must not recover a whole-circuit
            # partition through CommOpt's special full-span safeguard.
            cfg["hybrid_full_span_safeguard"] = False
            shared_ctx, prefix_time = self._build_layerwise_static_context(
                circuit,
                network,
                cfg,
            )
            prefix_time += self._assign_partition_plan(
                shared_ctx,
                partition_assigner_type="direct",
            )
            ctx = self._prepare_ctx_from_shared_prefix(shared_ctx, cfg)
            ctx.config["hybrid_full_span_safeguard"] = False
            records_start = time.time()
            pre_map_records = self._step_construct_hybrid_records_beam(ctx)
            records_time = time.time() - records_start
        else:
            shared_ctx, prefix_time = self._build_zero_cut_context(circuit, network, cfg)
            prefix_time += self._assign_partition_plan(shared_ctx)
            ctx = self._prepare_ctx_from_shared_prefix(shared_ctx, cfg)
            records_start = time.time()
            pre_map_records = self._step_construct_teledata_only_records(ctx)
            records_time = time.time() - records_start
            if any(record.mapping_type != "teledata" for record in pre_map_records.records):
                raise RuntimeError("SliceOpt+MapOpt must contain only TeleData records before MapOpt")

        if not pre_map_records.records:
            raise RuntimeError(f"{normalized_variant} produced no records before MapOpt")
        pre_map_records.ensure_record_ops(ctx.circuit, ctx.circuit_layers)
        pre_map_type_counts = {
            "teledata": sum(
                record.mapping_type == "teledata" for record in pre_map_records.records
            ),
            "telegate": sum(
                record.mapping_type in {"cat", "telegate"}
                for record in pre_map_records.records
            ),
        }
        mapper_id = self._resolve_mapopt_mapper_id(cfg, pre_map_records)
        result = self._map_records_with_mapper(
            ctx,
            pre_map_records,
            mapper_id=mapper_id,
            shared_prefix_time=prefix_time,
            records_time=records_time,
        )
        post_map_type_counts = {
            "teledata": sum(
                record.mapping_type == "teledata" for record in result.records
            ),
            "telegate": sum(
                record.mapping_type in {"cat", "telegate"}
                for record in result.records
            ),
        }
        stats = {
            "variant": normalized_variant,
            "slice_strategy": ctx.slice_strategy,
            "slice_planning_time_sec": float(ctx.slice_planning_time_sec),
            "segment_count": len(ctx.subc_ranges),
            "pre_map_record_count": len(pre_map_records.records),
            "pre_map_teledata_record_count": pre_map_type_counts["teledata"],
            "pre_map_telegate_record_count": pre_map_type_counts["telegate"],
            "post_map_record_count": len(result.records),
            "post_map_teledata_record_count": post_map_type_counts["teledata"],
            "post_map_telegate_record_count": post_map_type_counts["telegate"],
            "mapper_id": mapper_id,
            "telegate_total_calls": int(ctx.telegate_total_calls),
            "telegate_cache_hits": int(ctx.telegate_cache_hits),
            "telegate_cache_relaxed_hits": int(ctx.telegate_cache_relaxed_hits),
            "telegate_cache_misses": int(ctx.telegate_cache_misses),
            "full_span_safeguard_enabled": bool(
                ctx.config.get("hybrid_full_span_safeguard", True)
            ),
            "commopt_full_span_won": bool(
                ctx.commopt_full_span_won
                or (
                    len(pre_map_records.records) == 1
                    and pre_map_records.records[0].mapping_type in {"cat", "telegate"}
                    and int(pre_map_records.records[0].layer_start) == 0
                    and int(pre_map_records.records[0].layer_end)
                    == len(ctx.circuit_layers) - 1
                )
            ),
        }
        return result, stats

    def _build_zero_cut_context(
        self,
        circuit: QuantumCircuit,
        network: Network,
        config: Optional[dict[str, Any]] = None,
    ) -> tuple[CompilationContext, float]:
        if config is None:
            config = {}

        slice_strategy = self._resolve_slice_strategy(config)
        partitioner_type = config.get("partitioner", "recursive_dp")
        max_option = config.get("max_option", 1)
        partitioner = PartitionerFactory.create_partitioner(partitioner_type, network, max_options=max_option)

        ctx = CompilationContext(
            circuit=circuit,
            network=network,
            config=copy.deepcopy(config),
            partitioner=partitioner,
            slice_strategy=slice_strategy,
        )
        ctx.config["slice_strategy"] = slice_strategy

        start_time = time.time()

        self._step_remove_single_qubit_gates(ctx)
        ctx.P_table = self._step_build_partition_table(ctx)
        ctx.subc_ranges = self._step_select_slicing(ctx)

        minimum_raw = config.get("min_atomic_zero_cut_subcircuits", 0)
        try:
            minimum_atomic_subcircuits = int(minimum_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "min_atomic_zero_cut_subcircuits must be a nonnegative integer"
            ) from exc
        if minimum_atomic_subcircuits < 0:
            raise ValueError(
                "min_atomic_zero_cut_subcircuits must be a nonnegative integer"
            )

        atomic_subcircuit_count = len(ctx.subc_ranges)
        if (
            minimum_atomic_subcircuits > 0
            and atomic_subcircuit_count < minimum_atomic_subcircuits
        ):
            benchmark = str(config.get("benchmark", config.get("circuit_name", "unknown")))
            print(
                "[COMMOPT_INELIGIBLE] "
                + json.dumps({
                    "benchmark": benchmark,
                    "atomic_zero_cut_subcircuit_count": atomic_subcircuit_count,
                    "minimum_required": minimum_atomic_subcircuits,
                }, sort_keys=True),
                flush=True,
            )
            raise InsufficientAtomicZeroCutSubcircuitsError(
                benchmark,
                atomic_subcircuit_count,
                minimum_atomic_subcircuits,
            )

        ctx.partition_candidates = [ctx.P_table[i][j] for (i, j) in ctx.subc_ranges]

        return ctx, time.time() - start_time

    def _assign_partition_plan(
        self,
        ctx: CompilationContext,
        partition_assigner_type: Optional[str] = None,
    ) -> float:
        assigner_type = partition_assigner_type or str(
            ctx.config.get("partition_assigner", "max_match")
        )
        ctx.partition_assigner = PartitionAssignerFactory.create_assigner(assigner_type)
        assert ctx.partition_assigner is not None
        assign_time = time.time()
        assign_result = ctx.partition_assigner.assign(ctx.partition_candidates)
        ctx.partition_plan = assign_result["partition_plan"]
        log(
            f"[partition_assigner] type={assigner_type}, "
            f"name={ctx.partition_assigner.name}"
        )
        return time.time() - assign_time

    def compile_from_shared_prefix(
        self,
        shared_ctx: CompilationContext,
        config: Optional[dict[str, Any]] = None,
        shared_prefix_time: float = 0.0,
        override_mapper_id: Optional[str] = None,
    ) -> MappingRecordList:
        ctx = copy.deepcopy(shared_ctx)
        if config is None:
            config = {}
        ctx.config = copy.deepcopy(config)

        telegate_partitioner_type = config.get("telegate_partitioner", "cat")
        ctx.telegate_partitioner = TelegatePartitionerFactory.create_telegate_partitioner(telegate_partitioner_type)
        ctx.mapper = MapperFactory.create_mapper("direct")

        return self._finish_compile_from_ctx(
            ctx,
            shared_prefix_time=shared_prefix_time,
            override_mapper_id=override_mapper_id,
        )

    def _prepare_ctx_from_shared_prefix(
        self,
        shared_ctx: CompilationContext,
        config: Optional[dict[str, Any]] = None,
    ) -> CompilationContext:
        ctx = copy.deepcopy(shared_ctx)
        if config is None:
            config = {}
        ctx.config = copy.deepcopy(config)

        telegate_partitioner_type = config.get("telegate_partitioner", "cat")
        ctx.telegate_partitioner = TelegatePartitionerFactory.create_telegate_partitioner(telegate_partitioner_type)
        ctx.mapper = MapperFactory.create_mapper("direct")
        return ctx

    def compile_teledata_only_direct_from_shared_prefix(
        self,
        shared_ctx: CompilationContext,
        config: Optional[dict[str, Any]] = None,
        shared_prefix_time: float = 0.0,
    ) -> MappingRecordList:
        ctx = self._prepare_ctx_from_shared_prefix(shared_ctx, config)
        records_start_time = time.time()
        teledata_only_records = self._step_construct_teledata_only_records(ctx)
        records_time = time.time() - records_start_time
        return self._map_records_with_mapper(
            ctx,
            teledata_only_records,
            mapper_id="direct",
            shared_prefix_time=shared_prefix_time,
            records_time=records_time,
        )

    def _checkpoint_enabled(self, ctx: CompilationContext) -> bool:
        return bool(ctx.config.get("navi_checkpoint_enabled", False))

    def _checkpoint_dir(self, ctx: CompilationContext) -> Path:
        configured = ctx.config.get("navi_checkpoint_dir")
        if configured:
            return Path(str(configured))
        circuit_name = str(ctx.config.get("circuit_name", f"circuit_{ctx.circuit.num_qubits}"))
        return Path("outputs") / circuit_name / "navi_checkpoints"

    def _write_checkpoint_manifest_entry(
        self,
        checkpoint_dir: Path,
        key: str,
        filename: str,
        execution_time: Optional[float] = None,
    ) -> None:
        manifest_path = checkpoint_dir / "manifest.json"
        manifest: dict[str, Any] = {}
        if manifest_path.exists():
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                manifest = {str(k): v for k, v in loaded.items()}
        if execution_time is None:
            manifest[key] = filename
        else:
            manifest[key] = {
                "file": filename,
                "execution_time": float(execution_time),
            }
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=4, sort_keys=True),
            encoding="utf-8",
        )

    def _save_checkpoint_records(
        self,
        ctx: CompilationContext,
        key: str,
        filename: str,
        records: MappingRecordList,
        execution_time: Optional[float] = None,
    ) -> None:
        if not self._checkpoint_enabled(ctx):
            return

        checkpoint_dir = self._checkpoint_dir(ctx)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_records = copy.deepcopy(records)
        checkpoint_records.ensure_record_ops(ctx.circuit, ctx.circuit_layers)
        checkpoint_records.num_records = len(checkpoint_records.records)
        checkpoint_records.save_records(str(checkpoint_dir / filename), dump_type="raw")

        self._write_checkpoint_manifest_entry(
            checkpoint_dir,
            key,
            filename,
            execution_time=execution_time,
        )

    def _compile_impl(self,
                circuit: QuantumCircuit,
                network: Network,
                config: Optional[dict[str, Any]] = None,
                override_mapper_id: Optional[str] = None) -> MappingRecordList:
        """
        Compile the circuit using the NAVI algorithm.
        """
        print(f"Compiling with [{self.name}]...")
        print(f"Compiling with [{self.name}]...", file=sys.stderr)
        
        if config is None:
            config = {}

        if bool(config.get("ablation_mode", False)):
            return self._compile_ablation(circuit, network, config)

        shared_ctx, shared_prefix_time = self.build_shared_prefix_context(circuit, network, config)
        ctx = self._prepare_ctx_from_shared_prefix(shared_ctx, config)

        return self._finish_compile_from_ctx(
            ctx,
            shared_prefix_time=shared_prefix_time,
            override_mapper_id=override_mapper_id,
        )

    def _compile_ablation(
        self,
        circuit: QuantumCircuit,
        network: Network,
        config: dict[str, Any],
    ) -> MappingRecordList:
        """Build NAVI-Td, NAVI-Hy, and full NAVI from one zero-cut prefix."""
        if not bool(config.get("navi_checkpoint_enabled", False)):
            raise ValueError("NAVI ablation mode requires navi_checkpoint.enabled=true")

        zero_cut_ctx, zero_cut_time = self._build_zero_cut_context(
            circuit,
            network,
            config,
        )

        direct_assigned_ctx = copy.deepcopy(zero_cut_ctx)
        direct_assign_time = self._assign_partition_plan(
            direct_assigned_ctx,
            partition_assigner_type="direct",
        )

        td_ctx = self._prepare_ctx_from_shared_prefix(direct_assigned_ctx, config)
        td_start = time.time()
        teledata_records = self._step_construct_teledata_only_records(td_ctx)
        if any(record.mapping_type != "teledata" for record in teledata_records.records):
            raise RuntimeError("NAVI-Td ablation must contain only teledata records")
        teledata_records.ensure_record_ops(td_ctx.circuit, td_ctx.circuit_layers)
        td_time = zero_cut_time + direct_assign_time + (time.time() - td_start)
        self._save_checkpoint_records(
            td_ctx,
            "NAVI-Td",
            "NAVI-Td_raw.json",
            teledata_records,
            execution_time=td_time,
        )

        hy_ctx = self._prepare_ctx_from_shared_prefix(direct_assigned_ctx, config)
        hy_start = time.time()
        hybrid_records = self._step_construct_hybrid_records_beam(hy_ctx)
        hybrid_records.ensure_record_ops(hy_ctx.circuit, hy_ctx.circuit_layers)
        hy_time = zero_cut_time + direct_assign_time + (time.time() - hy_start)
        self._save_checkpoint_records(
            hy_ctx,
            "NAVI-Hy",
            "NAVI-Hy_raw.json",
            hybrid_records,
            execution_time=hy_time,
        )

        max_match_ctx = copy.deepcopy(zero_cut_ctx)
        max_match_time = self._assign_partition_plan(max_match_ctx)
        full_ctx = self._prepare_ctx_from_shared_prefix(max_match_ctx, config)
        hybrid_start = time.time()
        full_hybrid_records = self._step_construct_hybrid_records_beam(full_ctx)
        full_hybrid_records.ensure_record_ops(full_ctx.circuit, full_ctx.circuit_layers)
        hybrid_time = time.time() - hybrid_start

        mapper_id = str(
            config.get("multi_record_mapper", "boundeddp_neighbor")
        ).lower()
        if mapper_id == "boundeddp_neighbor" and len(full_hybrid_records.records) == 1:
            mapper_id = "single_record_greedy"

        return self._map_records_with_mapper(
            full_ctx,
            full_hybrid_records,
            mapper_id=mapper_id,
            shared_prefix_time=zero_cut_time + max_match_time,
            records_time=hybrid_time,
        )

    def _construct_records_from_ctx(
        self,
        ctx: CompilationContext,
        shared_prefix_time: float = 0.0,
    ) -> tuple[MappingRecordList, float]:
        records_start_time = time.time()

        teledata_only_records = self._step_construct_teledata_only_records(ctx)
        self._save_checkpoint_records(
            ctx,
            "NAVI-Td",
            "NAVI-Td_raw.json",
            teledata_only_records,
            execution_time=shared_prefix_time + (time.time() - records_start_time),
        )
        teledata_eval_epairs = float(teledata_only_records.total_costs.epairs)

        commopt_start = time.perf_counter()
        hybrid_records = self._step_construct_hybrid_records_beam(ctx)
        commopt_time = time.perf_counter() - commopt_start
        self._save_checkpoint_records(
            ctx,
            "NAVI-Hy",
            "NAVI-Hy_raw.json",
            hybrid_records,
            execution_time=shared_prefix_time + (time.time() - records_start_time),
        )
        hybrid_eval_epairs = hybrid_records.total_costs.epairs

        selected_records = hybrid_records
        selected_source = "hybrid"
        if teledata_eval_epairs < hybrid_eval_epairs:
            selected_records = teledata_only_records
            selected_source = "teledata"

        beam_width = max(1, int(ctx.config.get("hybrid_beam_width", 1)))
        max_merge_span = max(1, int(ctx.config.get("hybrid_max_merge_span", 10)))
        merge_probe_budget = max(1, int(ctx.config.get("hybrid_merge_probe_budget", 10)))
        pos_probe_budget = max(1, int(ctx.config.get("hybrid_pos_probe_budget", len(ctx.subc_ranges))))
        assigner_name = ctx.partition_assigner.name if ctx.partition_assigner is not None else "unknown"
        analysis_record = {
            "benchmark": str(ctx.config.get("benchmark", ctx.config.get("circuit_name", "unknown"))),
            "circuit_seed": ctx.config.get("circuit_seed"),
            "slice_strategy": ctx.slice_strategy,
            "slice_planning_time_sec": float(ctx.slice_planning_time_sec),
            "K": len(ctx.subc_ranges),
            "W": beam_width,
            "L_max": max_merge_span,
            "effective_local_Lmax": min(len(ctx.subc_ranges), max_merge_span),
            "partition_assigner": str(ctx.config.get("partition_assigner", "max_match")),
            "partition_assigner_name": assigner_name,
            "merge_probe_budget": merge_probe_budget,
            "pos_probe_budget": pos_probe_budget,
            "teledata_epairs_internal": float(teledata_eval_epairs),
            "hybrid_epairs_internal": float(hybrid_eval_epairs),
            "selected_epairs_internal": float(selected_records.total_costs.epairs),
            "selected_source": selected_source,
            "teledata_record_count": len(teledata_only_records.records),
            "hybrid_record_count": len(hybrid_records.records),
            "selected_record_count": len(selected_records.records),
            "commopt_time_sec": float(commopt_time),
            "telegate_total_calls": ctx.telegate_total_calls,
            "telegate_cache_hits": ctx.telegate_cache_hits,
            "telegate_cache_relaxed_hits": ctx.telegate_cache_relaxed_hits,
            "telegate_cache_misses": ctx.telegate_cache_misses,
            "full_span_safeguard_enabled": bool(
                ctx.config.get("hybrid_full_span_safeguard", True)
            ),
            "full_span_candidate_won": ctx.commopt_full_span_won,
        }
        print(f"[COMMOPT_ANALYSIS] {json.dumps(analysis_record, sort_keys=True)}")

        total_cache_hits = ctx.telegate_cache_hits + ctx.telegate_cache_relaxed_hits
        total_telegate_calls = total_cache_hits + ctx.telegate_cache_misses
        if total_telegate_calls > 0:
            hit_rate = total_cache_hits / total_telegate_calls
            strict_unique = len(ctx.telegate_strict_seen_keys)
            range_unique = len(ctx.telegate_range_seen_keys)
            strict_repeat = max(0, ctx.telegate_total_calls - strict_unique)
            range_repeat = max(0, ctx.telegate_total_calls - range_unique)
            log(
                f"[telegate_cache] strict_hits={ctx.telegate_cache_hits}, relaxed_hits={ctx.telegate_cache_relaxed_hits}, "
                f"misses={ctx.telegate_cache_misses}, "
                f"hit_rate={hit_rate:.2%}\n"
                f"[telegate_keys] calls={ctx.telegate_total_calls}, "
                f"strict_unique={strict_unique}, strict_repeat={strict_repeat}, "
                f"range_unique={range_unique}, range_repeat={range_repeat}"
            )

        return selected_records, time.time() - records_start_time

    def _map_records_with_mapper(
        self,
        ctx: CompilationContext,
        records: MappingRecordList,
        mapper_id: str,
        shared_prefix_time: float = 0.0,
        records_time: float = 0.0,
    ) -> MappingRecordList:
        mapper_key = str(mapper_id).lower()
        baseline_raw = copy.deepcopy(records)
        identity_perm = list(range(ctx.network.num_backends))
        for record in baseline_raw.records:
            extra_info = copy.deepcopy(record.extra_info) if record.extra_info is not None else {}
            extra_info["perm"] = identity_perm.copy()
            record.extra_info = extra_info
        baseline_raw.ensure_record_ops(ctx.circuit, ctx.circuit_layers)

        mapper_start_time = time.time()
        ctx.hybrid_records = copy.deepcopy(records)
        ctx.mapper = MapperFactory.create_mapper(mapper_key)

        candidate_internal = ctx.mapper.map(
            ctx.hybrid_records,
            ctx.circuit,
            ctx.circuit_layers,
            ctx.network,
            config=ctx.config,
        )

        candidate_internal.ensure_record_ops(ctx.circuit, ctx.circuit_layers)
        beam_stats = copy.deepcopy(
            getattr(candidate_internal, "mapopt_beam_stats", {}) or {}
        )
        candidate_raw = copy.deepcopy(candidate_internal)
        mapper_time = time.time() - mapper_start_time
        configured_output_base = ctx.config.get("record_output_base")
        if configured_output_base:
            base_path = str(Path(str(configured_output_base)))
        else:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            circuit_name = str(ctx.config.get("circuit_name", f"circuit_{ctx.circuit.num_qubits}"))
            base_path = CompilerUtils.build_record_base_path(
                circuit_name,
                ctx.network,
                self.name,
                timestamp,
                output_folder=str(ctx.config.get("_output_folder", "./outputs")),
            )

        policy_name = ctx.config.get("evaluator_policy")
        pre_eval_time = shared_prefix_time + records_time + mapper_time
        candidate_evaluated, _, candidate_eval_time = self._evaluate_raw_mapping_records_with_timing(
            copy.deepcopy(candidate_raw),
            ctx.network,
            policy_name=policy_name,
            config=ctx.config,
            pre_eval_time=pre_eval_time,
        )

        safeguard_enabled = (
            mapper_key != "direct"
            and bool(ctx.config.get("mapopt_dominance_safeguard", True))
        )
        cost_atol = float(ctx.config.get("mapopt_cost_atol", 1e-12))
        cost_rtol = float(ctx.config.get("mapopt_cost_rtol", 1e-12))
        if not math.isfinite(cost_atol) or cost_atol < 0:
            raise ValueError("mapopt_cost_atol must be a finite nonnegative number")
        if not math.isfinite(cost_rtol) or cost_rtol < 0:
            raise ValueError("mapopt_cost_rtol must be a finite nonnegative number")
        baseline_evaluated: Optional[MappingRecordList] = None
        baseline_eval_time = 0.0
        selected_source = "candidate"
        selection_reason = "safeguard_not_applicable"
        if safeguard_enabled:
            baseline_evaluated, _, baseline_eval_time = self._evaluate_raw_mapping_records_with_timing(
                copy.deepcopy(baseline_raw),
                ctx.network,
                policy_name=policy_name,
                config=ctx.config,
                pre_eval_time=pre_eval_time + candidate_eval_time,
            )
            selected_source, selection_reason = select_mapopt_source(
                baseline_evaluated.total_costs,
                candidate_evaluated.total_costs,
                atol=cost_atol,
                rtol=cost_rtol,
            )

        if selected_source == "no_op":
            assert baseline_evaluated is not None
            selected_raw = baseline_raw
            final_result = baseline_evaluated
        else:
            selected_raw = candidate_raw
            final_result = candidate_evaluated

        exec_time = pre_eval_time + candidate_eval_time + baseline_eval_time
        log(f"[INFO] {self.name} execution time: {exec_time} sec")

        final_result.update_total_costs(execution_time=exec_time)
        circuit_depth = int(ctx.circuit.depth())
        selection_summary = {
            "mapper_id": mapper_key,
            "selected_source": selected_source,
            "selection_reason": selection_reason,
            "safeguard_enabled": safeguard_enabled,
            "comparison_tolerances": {
                "mapopt_cost_atol": cost_atol,
                "mapopt_cost_rtol": cost_rtol,
            },
            "baseline": (
                _mapopt_metric_snapshot(baseline_evaluated.total_costs, circuit_depth)
                if baseline_evaluated is not None
                else None
            ),
            "candidate": _mapopt_metric_snapshot(
                candidate_evaluated.total_costs,
                circuit_depth,
            ),
            "beam_tie_statistics": beam_stats,
        }

        selected_raw.save_records(f"{base_path}_raw.json", dump_type="raw")
        final_result.save_records(f"{base_path}_evaluated.json", dump_type="evaluated")
        _atomic_write_json(
            Path(f"{base_path}_mapopt_selection.json"),
            selection_summary,
        )
        if selected_source == "no_op":
            candidate_raw.save_records(
                f"{base_path}_mapopt_candidate_raw.json",
                dump_type="raw",
            )
            candidate_evaluated.save_records(
                f"{base_path}_mapopt_candidate_evaluated.json",
                dump_type="evaluated",
            )

        if self._checkpoint_enabled(ctx):
            self._save_checkpoint_records(
                ctx,
                "NAVI",
                "NAVI_raw.json",
                selected_raw,
                execution_time=exec_time,
            )
            checkpoint_dir = self._checkpoint_dir(ctx)
            final_result.save_records(
                str(checkpoint_dir / "NAVI_evaluated.json"),
                dump_type="evaluated",
            )
            self._write_checkpoint_manifest_entry(
                checkpoint_dir,
                "NAVI-evaluated",
                "NAVI_evaluated.json",
            )
            _atomic_write_json(
                checkpoint_dir / "NAVI_mapopt_selection.json",
                selection_summary,
            )
            self._write_checkpoint_manifest_entry(
                checkpoint_dir,
                "NAVI-MapOpt-Selection",
                "NAVI_mapopt_selection.json",
            )
            if selected_source == "no_op":
                candidate_raw.save_records(
                    str(checkpoint_dir / "NAVI_mapopt_candidate_raw.json"),
                    dump_type="raw",
                )
                candidate_evaluated.save_records(
                    str(checkpoint_dir / "NAVI_mapopt_candidate_evaluated.json"),
                    dump_type="evaluated",
                )
                self._write_checkpoint_manifest_entry(
                    checkpoint_dir,
                    "NAVI-MapOpt-Candidate-Raw",
                    "NAVI_mapopt_candidate_raw.json",
                )
                self._write_checkpoint_manifest_entry(
                    checkpoint_dir,
                    "NAVI-MapOpt-Candidate-Evaluated",
                    "NAVI_mapopt_candidate_evaluated.json",
                )
        return final_result

    def _finish_compile_from_ctx(
        self,
        ctx: CompilationContext,
        shared_prefix_time: float = 0.0,
        override_mapper_id: Optional[str] = None,
    ) -> MappingRecordList:
        selected_records, records_time = self._construct_records_from_ctx(
            ctx,
            shared_prefix_time=shared_prefix_time,
        )
        if override_mapper_id is not None:
            selected_mapper_id = str(override_mapper_id).lower()
        elif ctx.network.net_type == "all_to_all":
            # MapOpt 对 all-to-all 网络影响很小，scalability 测试中跳过排列搜索。
            selected_mapper_id = "direct"
        else:
            selected_mapper_id = str(ctx.config.get("multi_record_mapper", "boundeddp_neighbor")).lower()
            if (
                selected_mapper_id == "boundeddp_neighbor"
                and len(selected_records.records) == 1
            ):
                selected_mapper_id = "single_record_greedy"
        return self._map_records_with_mapper(
            ctx,
            selected_records,
            mapper_id=selected_mapper_id,
            shared_prefix_time=shared_prefix_time,
            records_time=records_time,
        )

    # =========================================================================
    # 步骤实现 (Step Implementations)
    # =========================================================================

    def _sample_budgeted_indices(self, start_idx: int, end_idx: int, budget: int) -> list[int]:
        """
        在[start_idx, end_idx]范围内按预算采样索引，保证包含头尾。
        """
        if end_idx < start_idx:
            return []

        budget = max(1, int(budget))
        span_len = end_idx - start_idx + 1
        if span_len <= budget:
            return list(range(start_idx, end_idx + 1))

        picked: set[int] = {start_idx, end_idx}
        if budget > 2:
            inner = budget - 2
            for t in range(1, inner + 1):
                idx = start_idx + (t * (span_len - 1)) // (inner + 1)
                picked.add(idx)

        return sorted(picked)

    def _step_construct_teledata_only_records(self, ctx: CompilationContext) -> MappingRecordList:
        """
        基于partition_plan构建teledata-only记录，并将all-to-all口径epairs写入record.costs。
        """
        mapping_record_list = MappingRecordList()
        eval_network = ctx.network
        prev_partition: Optional[list[list[int]]] = None

        for idx, (s, e) in enumerate(ctx.subc_ranges):
            left, right = self.get_original_layer_idx(ctx, (s, e))
            record = MappingRecord(
                layer_start=left,
                layer_end=right,
                partition=ctx.partition_plan[idx],
                mapping_type="teledata",
            )

            if prev_partition is not None:
                td_costs, _ = CompilerUtils.evaluate_teledata(prev_partition, record.partition, eval_network)
                record.costs = copy.deepcopy(td_costs)
            else:
                record.costs = ExecCosts()
            prev_partition = record.partition

            mapping_record_list.add_record(record)

        mapping_record_list.summarize_total_costs()
        return mapping_record_list

    def _step_remove_single_qubit_gates(self, ctx: CompilationContext):
        """
        Remove single qubit gates and calculate densities.
        Updates: ctx.multiq_layers, ctx.map_*, ctx.gate_density, ctx.twoq_gate_density
        """
        ctx.dag = circuit_to_dag(ctx.circuit)
        
        circuit_layers = [] # 每一层存放的是原始量子线路上所有量子门，如果双量子比特门被拆分了，它也要被拆分
        multiq_layers = []
        map_to_circuit_layer = {}
        pos_count, twoq_count, gate_count = 0, 0, 0

        n_backends = ctx.network.num_backends
        
        layers = list(ctx.dag.layers())
        for lev, layer in enumerate(layers):
            twoq_gates = []
            all_gates = []

            for node in layer["graph"].op_nodes():
                if not hasattr(node, 'qargs') or node.qargs is None:
                    raise ValueError("[ERROR] node.qargs does not exist or is none")
                pos_count += len(node.qargs)
                gate_count += 1
                
                if len(node.qargs) > 1:
                    if node.op.name == "barrier":
                        continue
                    if len(node.qargs) != 2:
                        raise ValueError(f"[ERROR] Found gate with more than 2 qubits: {node.op.name} on {node.qargs}")
                    twoq_count += 1
                    twoq_gates.append(node)
                else:
                    all_gates.append(node)

            if len(twoq_gates) > 0: # 逻辑：如果层满且不能整除后端数，则拆分
                if len(twoq_gates) == ctx.circuit.num_qubits // 2 and len(twoq_gates) % n_backends != 0:
                    split_point = len(twoq_gates) // 2
                    halves = [twoq_gates[:split_point], twoq_gates[split_point:]]

                    # 前一半，直接extend到all_gates当中，加入circuit_layers
                    all_gates.extend(halves[0]) # 单量子比特门 + 前一半双量子比特门
                    circuit_layers.append(all_gates)
                    multiq_layers.append(halves[0])
                    map_to_circuit_layer[len(multiq_layers) - 1] = len(circuit_layers) - 1

                    # 后一半，作为新的一层，加入circuit_layers
                    circuit_layers.append(halves[1]) # 后一半双量子比特门作为新的一层
                    multiq_layers.append(halves[1])
                    map_to_circuit_layer[len(multiq_layers) - 1] = len(circuit_layers) - 1
                else:
                    all_gates.extend(twoq_gates) # 单量子比特门 + 双量子比特门
                    circuit_layers.append(all_gates)
                    multiq_layers.append(twoq_gates)
                    map_to_circuit_layer[len(multiq_layers) - 1] = len(circuit_layers) - 1
            elif len(all_gates) > 0: # 不是空层
                # 没有双量子比特门的层，直接放到circuit_layers里
                circuit_layers.append(all_gates)
        
        assert 0 in map_to_circuit_layer, "[ERROR] map_to_circuit_layer must contain the first layer mapping."

        # 计算密度
        depth = ctx.circuit.depth()
        num_q = ctx.circuit.num_qubits
        ctx.gate_density = pos_count / (num_q * depth) if depth > 0 and num_q > 0 else 0.0
        ctx.twoq_gate_density = twoq_count / gate_count if gate_count > 0 else 0.0
        
        # 更新 Context
        ctx.circuit_layers = circuit_layers
        ctx.multiq_layers = multiq_layers
        ctx.map_to_circuit_layer = map_to_circuit_layer

        return

    def _step_build_partition_table(self, ctx: CompilationContext) -> list[Any]:
        """
        Build the P table.
        """
        multiq_layers = ctx.multiq_layers
        num_depths = len(multiq_layers)
        
        if num_depths == 0:
            return []

        P = [[[] for _ in range(num_depths)] for _ in range(num_depths)]
        cnt = 0
        
        qig = self._build_qubit_interaction_graph_by_level(ctx, (0, num_depths-1))
        is_changed = True

        assert ctx.partitioner is not None

        for i in range(num_depths):
            # ===== P[i][numDepths-1] =====
            if i != 0:
                is_changed = self._remove_qig_edge(ctx, qig, i-1, multiq_layers)

            if len(P[i][num_depths-1]) > 0:
                if i == 0:
                    raise RuntimeError(f"[ERROR] P[{i}][{num_depths-1}] should be empty.")
                success = True # leftward propagation
                if i + 1 < num_depths: # downward propagation
                    P[i+1][num_depths-1] = P[i][num_depths-1]
            else:
                success = False
                if is_changed:
                    P[i][num_depths-1] = self._get_qig_partitions(qig, ctx.partitioner)
                    cnt += 1
                    if len(P[i][num_depths-1]) > 0:
                        success = True # leftward propagation
                        if i + 1 < num_depths:
                            P[i+1][num_depths-1] = P[i][num_depths-1]
        
            # ===== P[i][numDepths-2 ~ i] =====
            qig_tmp = qig.copy()
            for j in range(num_depths - 2, i - 1, -1):
                is_changed = self._remove_qig_edge(ctx, qig_tmp, j+1, multiq_layers)
                
                if len(P[i][j]) > 0: # inherit from the upper grid
                    success = True # leftward propagation
                    if i + 1 <= j:
                        P[i+1][j] = P[i][j] # downward propagation
                elif success: # inherit from the right grid
                    P[i][j] = P[i][j+1]
                else:
                    if is_changed:
                        P[i][j] = self._get_qig_partitions(qig_tmp, ctx.partitioner)
                        cnt += 1
                        if len(P[i][j]) > 0:
                            success = True # leftward propagation
                            if i + 1 <= j:
                                P[i+1][j] = P[i][j]

        log(f"[build_partition_table] Partition calculation times: {cnt}.")

        return P

    def _step_select_slicing(self, ctx: CompilationContext) -> list[tuple[int, int]]:
        """Select the configured slicing strategy over the shared P table."""
        strategy = self._resolve_slice_strategy(ctx.config)
        start_time = time.perf_counter()

        if strategy == "dp":
            subc_ranges = self._step_build_slicing_table(ctx)
        elif strategy == "greedy_table":
            subc_ranges = self._step_build_greedy_slicing(ctx)
        else:  # Guard against future registry/config drift.
            raise AssertionError(f"Unhandled slice strategy: {strategy}")

        ctx.slice_strategy = strategy
        ctx.slice_planning_time_sec = time.perf_counter() - start_time
        log(
            f"[slice_planner] strategy={strategy}, layers={len(ctx.P_table)}, "
            f"segments={len(subc_ranges)}, "
            f"time_sec={ctx.slice_planning_time_sec:.6f}"
        )
        return subc_ranges

    def _step_build_greedy_slicing(
        self,
        ctx: CompilationContext,
    ) -> list[tuple[int, int]]:
        """Choose the longest zero-cut-feasible prefix at each slice boundary."""
        P = ctx.P_table
        num_depths = len(P)
        if num_depths == 0:
            return []

        for i in range(num_depths):
            if len(P[i][i]) == 0:
                raise RuntimeError(
                    f"[ERROR] P[{i}][{i}] is empty. Cannot build greedy slicing."
                )

        subc_ranges: list[tuple[int, int]] = []
        i = 0
        while i < num_depths:
            r = i
            while r + 1 < num_depths and len(P[i][r + 1]) > 0:
                r += 1
            subc_ranges.append((i, r))
            i = r + 1

        next_left = 0
        for left, right in subc_ranges:
            assert left == next_left, "Greedy slicing must be contiguous."
            assert left <= right < num_depths, "Greedy slicing range is invalid."
            assert len(P[left][right]) > 0, "Greedy slicing selected an infeasible range."
            assert (
                right == num_depths - 1 or len(P[left][right + 1]) == 0
            ), "Greedy slicing range is not right-maximal."
            next_left = right + 1
        assert next_left == num_depths, "Greedy slicing must cover every layer."

        return subc_ranges

    def _step_build_slicing_table(self, ctx: CompilationContext) -> list[Any]:
        """
        Build S and T tables and extract subc_ranges.
        """
        P = ctx.P_table
        num_depths = len(P)

        if num_depths == 0:
            return []

        T = [[0] * num_depths for _ in range(num_depths)]
        S = [[-1] * num_depths for _ in range(num_depths)]

        for i in range(num_depths):
            if len(P[i][i]) == 0:
                # 输出当前层上的量子操作
                for node in ctx.multiq_layers[i]:
                    print(f"[DEBUG] node: {node}, {node.op.name}, {node.qargs}", file=sys.stderr)
                raise RuntimeError(f"[ERROR] P[{i}][{i}] is empty. Cannot build slicing table.")

        for depth in range(2, num_depths + 1):
            for i in range(0, num_depths - depth + 1): # 左边界
                j = i + depth - 1 # 右边界
                if len(P[i][j]) == 0: # 需要切分
                    lower_k = S[i][j-1] if S[i][j-1] != -1 else i
                    upper_k = S[i+1][j] if S[i+1][j] != -1 else j-1
                    
                    best_k = -1
                    min_val = 10 ** 100
                    for k in range(lower_k, upper_k + 1):
                        comms = T[i][k] + T[k+1][j] + 1
                        if comms < min_val:
                            min_val = comms
                            best_k = k
                    T[i][j] = min_val
                    S[i][j] = best_k

        # Extract ranges
        ctx.S_table = S
        ctx.T_table = T

        subc_ranges = []
        self._get_sliced_subc_recursive(S, 0, num_depths - 1, subc_ranges)
        
        return subc_ranges

    def _step_construct_hybrid_records_beam(self, ctx: CompilationContext) -> MappingRecordList:
        """
        束搜索构建混合记录

        配置项：
        - hybrid_beam_width: 每个位置保留的候选数
        - hybrid_max_merge_span: 单次 telegate 最长覆盖原子块数
        """
        blocks = ctx.subc_ranges
        K = len(blocks)
        ctx.commopt_full_span_won = False

        beam_width = int(ctx.config.get("hybrid_beam_width", 1))
        # 最多合并多少个block
        max_merge_span = int(ctx.config.get("hybrid_max_merge_span", 10))
        # 在每个pos，最多尝试几次_try_generate_telegate
        merge_probe_budget = int(ctx.config.get("hybrid_merge_probe_budget", 10))
        # 最多探查几个pos
        pos_probe_budget = int(ctx.config.get("hybrid_pos_probe_budget", K))
        full_span_safeguard = bool(
            ctx.config.get("hybrid_full_span_safeguard", True)
        )
        # 仅当候选显著超出 beam_width 时才做前向剪枝，减少重复排序开销。
        prune_trigger_ratio = int(ctx.config.get("hybrid_prune_trigger_ratio", 5))
        # 防止数值输入有误
        beam_width = max(1, beam_width)
        max_merge_span = max(1, max_merge_span)
        merge_probe_budget = max(1, merge_probe_budget)
        pos_probe_budget = max(1, pos_probe_budget)
        prune_trigger_ratio = max(2, prune_trigger_ratio)

        setattr(ctx, "runtime_hybrid_beam_width", beam_width)

        if K == 0:
            return MappingRecordList()

        eval_network = ctx.network
        log(
            "[hybrid] CAT initial_partition=previous_record_partition, "
            f"full_span_safeguard={full_span_safeguard}"
        )

        original_records: list[MappingRecord] = []
        for idx, (s, e) in enumerate(blocks):
            left, right = self.get_original_layer_idx(ctx, (s, e))
            record = MappingRecord(
                layer_start=left,
                layer_end=right,
                partition=ctx.partition_plan[idx],
                mapping_type="teledata",
            )
            original_records.append(record)

        # states[pos] = [(projected_epairs, projected_cat_ents, [records...]), ...]
        states: dict[int, list[tuple[float, int, list[MappingRecord]]]] = {0: [(0.0, 0, [])]}
        # 整段 [0, K-1] 直接 telegate这一条完整候选
        full_span_telegate_candidate: Optional[tuple[tuple[float, float], MappingRecordList]] = None
        def _state_key(item: tuple[float, int, list[MappingRecord]]) -> tuple[float, int]:
            projected_epairs, projected_cat_ents, _ = item
            return (projected_epairs, -projected_cat_ents)

        def _reevaluate_records(records: list[MappingRecord]) -> tuple[MappingRecordList, tuple[float, float]]:
            reevaluated = MappingRecordList(records=copy.deepcopy(records))
            logical_phy_map: dict[int, tuple[int, int | None]] = {}
            comm_phy_map: dict[int, list[int | None]] = {}

            for t, record in enumerate(reevaluated.records):
                if t == 0:
                    logical_phy_map = CompilerUtils.init_logical_phy_map(record.partition)
                    comm_phy_map = {}
                else:
                    logical_phy_map = copy.deepcopy(reevaluated.records[t - 1].logical_phy_map)
                    comm_phy_map = copy.deepcopy(getattr(reevaluated.records[t - 1], "comm_phy_map", {}) or {})

                record.costs = ExecCosts()
                record.logical_phy_map = copy.deepcopy(logical_phy_map)
                record.comm_phy_map = copy.deepcopy(comm_phy_map)

                if record.extra_info is not None and "ops" in record.extra_info:
                    subcircuit = record.extra_info["ops"]
                else:
                    subcircuit = CompilerUtils.get_subcircuit_by_level(
                        num_qubits=ctx.circuit.num_qubits,
                        circuit=ctx.circuit,
                        circuit_layers=ctx.circuit_layers,
                        layer_start=record.layer_start,
                        layer_end=record.layer_end,
                    )

                if t != 0:
                    prev_record = reevaluated.records[t - 1]
                    _, logical_phy_map, comm_phy_map = CompilerUtils.evaluate_teledata_with_local(
                        prev_record,
                        record,
                        ctx.network,
                    )

                CompilerUtils.evaluate_local_and_telegate_with_cat(
                    record,
                    subcircuit,
                    ctx.network,
                    config=ctx.config,
                )

            reevaluated.summarize_total_costs()
            total_costs = reevaluated.total_costs
            return reevaluated, (float(total_costs.total_fidelity_loss), float(total_costs.epairs))

        def _partition_sig(partition: list[list[int]]) -> tuple[tuple[int, ...], ...]:
            return tuple(tuple(sorted(part)) for part in partition)

        def _prune(candidates: list[tuple[float, int, list[MappingRecord]]]) -> list[tuple[float, int, list[MappingRecord]]]:
            # 先按“末尾划分签名”去重，保留每种签名成本最小者，再按总成本截断
            best_by_sig: dict[tuple[tuple[int, ...], ...], tuple[float, int, list[MappingRecord]]] = {}
            for item in candidates:
                _, _, recs = item
                if not recs:
                    sig = tuple()
                else:
                    sig = _partition_sig(recs[-1].partition)

                old = best_by_sig.get(sig)
                if old is None or _state_key(item) < _state_key(old):
                    best_by_sig[sig] = item

            pruned = sorted(best_by_sig.values(), key=_state_key)
            return pruned[:beam_width]

        telegate_probe_positions = set(self._sample_budgeted_indices(0, K - 1, pos_probe_budget))

        for pos in range(K):
            # 消费前剪枝：保证本层展开输入受控。
            if pos in states and len(states[pos]) > beam_width:
                states[pos] = _prune(states[pos])
            current_states = states.get(pos, [])

            if not current_states:
                log(f"[WARNING] pos {pos} has no states")
                continue

            for base_epairs, base_cat_ents, base_records in current_states:
                prev_partition = base_records[-1].partition if base_records else None

                # 选项 A：当前原子块保持 teledata
                td_record = copy.deepcopy(original_records[pos])
                if prev_partition is not None:
                    td_costs, _ = CompilerUtils.evaluate_teledata(
                        prev_partition, td_record.partition, eval_network
                    )
                    td_record.costs = copy.deepcopy(td_costs)
                else:
                    td_record.costs = ExecCosts()

                transition_cost = float(td_record.costs.epairs)

                next_epairs = base_epairs + transition_cost
                states.setdefault(pos + 1, []).append((next_epairs, base_cat_ents, base_records + [td_record]))

                # 选项 B：仅在采样到的pos上展开telegate，控制重计算规模。
                if pos in telegate_probe_positions:
                    start_layer = original_records[pos].layer_start
                    upper_end = min(K - 1, pos + max_merge_span - 1)
                    sampled_ends = self._sample_budgeted_indices(pos, upper_end, merge_probe_budget)

                    if (
                        full_span_safeguard
                        and pos == 0
                        and (K - 1) not in sampled_ends
                    ):
                        sampled_ends.append(K - 1)
                        sampled_ends = sorted(set(sampled_ends))

                    for end in sampled_ends:
                        if end == pos:
                            continue

                        end_layer = original_records[end].layer_end
                        is_full_span = pos == 0 and end == K - 1
                        telegate_result = self._try_generate_telegate(
                            ctx,
                            start_layer,
                            end_layer,
                            prev_partition=prev_partition,
                        )

                        tg_record = copy.deepcopy(telegate_result.records[0])
                        if prev_partition is not None:
                            tg_link_costs, _ = CompilerUtils.evaluate_teledata(
                                prev_partition, tg_record.partition, eval_network
                            )
                            tg_record.costs += tg_link_costs

                        seg_cat_ents = int(telegate_result.total_costs.cat_ents)
                        merged_epairs = base_epairs + float(tg_record.costs.epairs)
                        merged_cat_ents = base_cat_ents + seg_cat_ents

                        candidate_records = base_records + [tg_record]
                        states.setdefault(end + 1, []).append(
                            (merged_epairs, merged_cat_ents, candidate_records)
                        )

                        if is_full_span and full_span_safeguard:
                            full_span_result, full_span_eval_key = _reevaluate_records(
                                [tg_record]
                            )
                            full_span_telegate_candidate = (
                                full_span_eval_key,
                                full_span_result,
                            )

            # 修剪不好的records路径，控制状态爆炸
            for prune_pos in range(pos + 1, min(K, pos + max_merge_span) + 1):
                if prune_pos in states and len(states[prune_pos]) > prune_trigger_ratio * beam_width:
                    states[prune_pos] = _prune(states[prune_pos])
            
        final_candidates = states.get(K, [])
        if len(final_candidates) > beam_width:
            final_candidates = _prune(final_candidates)
        if not final_candidates:
            raise RuntimeError("[ERROR] beam search failed to produce any hybrid plan.")

        _, _, best_records = min(final_candidates, key=_state_key)
        result, best_eval_key = _reevaluate_records(best_records)

        if full_span_telegate_candidate is not None and full_span_telegate_candidate[0] < best_eval_key:
            best_eval_key, best_full_span_telegate_result = full_span_telegate_candidate
            result = copy.deepcopy(best_full_span_telegate_result)

        if K > 1 and len(result.records) == 1:
            only_record = result.records[0]
            ctx.commopt_full_span_won = (
                only_record.mapping_type in {"cat", "telegate"}
                and only_record.layer_start == original_records[0].layer_start
                and only_record.layer_end == original_records[-1].layer_end
            )

        ctx.telegate_optimized = True

        return result

    # =========================================================================
    # 底层逻辑辅助函数 (Helper Functions)
    # =========================================================================

    # P_table
    def _build_qubit_interaction_graph_by_level(self, ctx: CompilationContext, level_range: tuple[int, int]) -> nx.Graph:
        G = nx.Graph()
        for qubit in range(ctx.circuit.num_qubits):
            G.add_node(qubit)
        
        for lev in range(level_range[0], level_range[1]+1):
            for node in ctx.multiq_layers[lev]:
                qubits = []
                for q in node.qargs:
                    qubits.append(ctx.circuit.qubits.index(q))

                assert len(qubits) == 2

                q0, q1 = qubits
                if G.has_edge(q0, q1):
                    G[q0][q1]['weight'] += 1
                else:
                    G.add_edge(q0, q1, weight=1)
        return G

    def _remove_qig_edge(self, 
                         ctx: CompilationContext, 
                         qig: nx.Graph, 
                         lev: int, 
                         multiq_layers: list) -> bool:
        is_changed = False
        for node in multiq_layers[lev]:
            qubits = []
            for q in node.qargs:
                qubits.append(ctx.circuit.qubits.index(q))

            assert len(qubits) == 2

            q0, q1 = qubits
            if qig.has_edge(q0, q1):
                qig[q0][q1]['weight'] -= 1
                if qig[q0][q1]['weight'] == 0:
                    qig.remove_edge(q0, q1)
                    is_changed = True
        return is_changed

    def _get_qig_partitions(self, qig: nx.Graph, partitioner: Partitioner):
        components = [list(comp) for comp in nx.connected_components(qig)]
        return partitioner.partition(components)

    # S_table
    def _get_sliced_subc_recursive(self, S: list[list[int]], i: int, j: int, result_list: list[tuple[int, int]]):
        if S[i][j] == -1:
            result_list.append((i, j))
            return
        self._get_sliced_subc_recursive(S, i, S[i][j], result_list)
        self._get_sliced_subc_recursive(S, S[i][j] + 1, j, result_list)

    def _get_ori_subc(self, ctx: CompilationContext, ori_left: int, ori_right: int) -> QuantumCircuit:
        """
        获取原线路中[ori_left, ori_right]的子线路
        """
        # 1. 复制原电路的寄存器结构（包含量子和经典比特）
        # 这样可以完美保留原电路的比特命名和结构
        qregs = ctx.circuit.qregs
        cregs = ctx.circuit.cregs
        subcircuit = QuantumCircuit(*qregs, *cregs)

        # 2. 建立映射：{原电路比特对象: 子电路对应索引的比特对象}
        # 量子比特映射
        q_map = {old_q: subcircuit.qubits[i] for i, old_q in enumerate(ctx.circuit.qubits)}
        # 经典比特映射（防止有测量门等操作经典比特的情况）
        c_map = {old_c: subcircuit.clbits[i] for i, old_c in enumerate(ctx.circuit.clbits)} if subcircuit.clbits else {}

        # 3. 遍历并追加门，使用映射后的比特
        for lev in range(ori_left, ori_right + 1):
            for node in ctx.circuit_layers[lev]:
                # 替换量子比特
                new_qargs = [q_map[q] for q in node.qargs]
                # 替换经典比特（如果有的话）
                old_cargs = getattr(node, 'cargs', [])
                new_cargs = [c_map[c] for c in old_cargs] if old_cargs else []

                subcircuit.append(node.op, new_qargs, new_cargs)

        subcircuit = transpile(subcircuit, optimization_level=0)
        return subcircuit

    def _try_generate_telegate(
            self, 
            ctx: CompilationContext, 
            ori_left: int, ori_right: int,
            prev_partition: Optional[list[list]] = None) -> MappingRecordList:
        """
        尝试将 multiq_layer 区间 [s_multi, e_multi] 作为一个整体生成 telegate。
        返回 (record_list, cost) 或 None。
        """
        def _partition_signature(partition: Optional[list[list]]) -> tuple[Any, ...]:
            if partition is None:
                return ("NONE",)
            # 外层顺序保持不变（对应后端索引），内层排序避免同构列表顺序差异导致 miss。
            return tuple(tuple(sorted(part)) for part in partition)

        use_cache = bool(ctx.config.get("telegate_cache", True))
        runtime_beam_width = int(getattr(ctx, "runtime_hybrid_beam_width", 1))
        if runtime_beam_width == 1:
            use_cache = False
        strict_key = None
        range_key = None
        cache_mode = "strict"

        ctx.telegate_total_calls += 1

        if use_cache:
            cache_mode = str(ctx.config.get("telegate_cache_mode", "range"))
            range_key = (ori_left, ori_right, "ANY_PREV")
            use_strict_key = (cache_mode != "range")
            if use_strict_key:
                strict_key = (ori_left, ori_right, _partition_signature(prev_partition))

            if strict_key is not None:
                ctx.telegate_strict_seen_keys.add(strict_key)
            ctx.telegate_range_seen_keys.add(range_key)

            if cache_mode == "range":
                assert range_key is not None
                if range_key in ctx.telegate_cache:
                    ctx.telegate_cache_relaxed_hits += 1
                    return copy.deepcopy(ctx.telegate_cache[range_key])
            else:
                # strict / range_fallback / unknown(按strict处理)
                assert strict_key is not None
                if strict_key in ctx.telegate_cache:
                    ctx.telegate_cache_hits += 1
                    return copy.deepcopy(ctx.telegate_cache[strict_key])

                assert range_key is not None
                if cache_mode == "range_fallback" and range_key in ctx.telegate_cache:
                    ctx.telegate_cache_relaxed_hits += 1
                    return copy.deepcopy(ctx.telegate_cache[range_key])

            ctx.telegate_cache_misses += 1
        
        # 1. 提取子电路
        sub_qc = self._get_ori_subc(ctx, ori_left, ori_right)

        # 2. 调用 partitioner
        eval_network = ctx.network
        assert ctx.telegate_partitioner is not None
        telegate_record = ctx.telegate_partitioner.partition(
            circuit = sub_qc,
            network = eval_network,
            config = {
                "layer_start": ori_left,
                "layer_end": ori_right,
                "partition": prev_partition,
                "use_oee_init": bool(ctx.config.get("use_oee_init", True)),
                "iteration": int(ctx.config.get("iteration", 50)),
                "add_teledata_costs": False,
            }
        )

        if isinstance(telegate_record, MappingRecordList):
            record_list = telegate_record
        else:
            if telegate_record.extra_info is None:
                telegate_record.extra_info = {}

            record_list = MappingRecordList()
            record_list.add_record(telegate_record)
            record_list.summarize_total_costs()

        self._set_simple_telegate_ops_for_full_circuit_record(ctx, record_list, ori_left, ori_right)

        if use_cache:
            # strict模式只写strict key；range模式只写range key；fallback模式双写提高复用。
            if cache_mode == "range":
                assert range_key is not None
                ctx.telegate_cache[range_key] = copy.deepcopy(record_list)
            elif cache_mode == "range_fallback":
                assert strict_key is not None
                assert range_key is not None
                cached = copy.deepcopy(record_list)
                ctx.telegate_cache[strict_key] = cached
                ctx.telegate_cache[range_key] = copy.deepcopy(cached)
            else:
                assert strict_key is not None
                ctx.telegate_cache[strict_key] = copy.deepcopy(record_list)
        
        return record_list

    def _set_simple_telegate_ops_for_full_circuit_record(
        self,
        ctx: CompilationContext,
        record_list: MappingRecordList,
        ori_left: int,
        ori_right: int,
    ) -> None:
        is_full_circuit_range = (
            int(ori_left) == 0
            and int(ori_right) == len(ctx.circuit_layers) - 1
        )
        for record in record_list.records:
            if record.extra_info is None:
                record.extra_info = {}
            if is_full_circuit_range:
                static_oee_view_layers = CompilerUtils.build_circuit_layers(ctx.circuit)
                record.extra_info["simple_telegate_ops"] = CompilerUtils.get_subcircuit_by_level(
                    num_qubits=ctx.circuit.num_qubits,
                    circuit=ctx.circuit,
                    circuit_layers=static_oee_view_layers,
                    layer_start=0,
                    layer_end=len(static_oee_view_layers) - 1,
                )
            else:
                record.extra_info.pop("simple_telegate_ops", None)

    # Others
    def reconstruct_and_visualize_circuit(self, ctx: CompilationContext, verbose: bool = True) -> QuantumCircuit:
        """
        Reconstructs the multi-qubit gate circuit from the context's multiq_layers and visualizes it.
        
        Args:
            ctx (CompilationContext): The compilation context containing multiq_layers and original circuit info.
            verbose (bool): If True, prints the visualization and statistics to stdout.
            
        Returns:
            QuantumCircuit: The reconstructed circuit containing only multi-qubit gates.
        """
        multiq_layers = ctx.multiq_layers
        
        # 1. 检查数据有效性
        if not multiq_layers:
            if verbose:
                print("[INFO] No multi-qubit gates found in context. Cannot visualize circuit.")
            # 返回一个空的量子线路，保持返回值类型一致
            return QuantumCircuit(ctx.circuit.num_qubits)

        # 2. 调试信息 (可选)
        if verbose:
            print(f"[DEBUG] Reconstructing circuit from {len(multiq_layers)} layers...")

        # 3. 创建新的量子线路
        n_qubits = ctx.circuit.num_qubits
        recon_circ = QuantumCircuit(n_qubits)
        
        total_gates = 0
        # 4. 按层添加双量子门
        for i, layer in enumerate(multiq_layers):
            for node in layer:
                # 安全性检查：确保 node 有必要的属性
                if hasattr(node, 'op') and hasattr(node, 'qargs'):
                    recon_circ.append(node.op, node.qargs, getattr(node, 'cargs', []))
                    total_gates += 1
                else:
                    print(f"[WARNING] Skipping invalid node in layer {i}: {node}")

            # 在每层后添加 barrier（最后一层不加）
            if i < len(multiq_layers) - 1:
                recon_circ.barrier()

        # 5. 打印可视化结果
        if verbose:
            print("\n" + "="*60)
            print("Reconstructed Multi-Qubit Gate Circuit Visualization")
            print("="*60)
            print(recon_circ) # Qiskit 会自动调用 circuit_drawer
            print("="*60)
            print(f"Statistics:")
            print(f"  - Total Layers: {len(multiq_layers)}")
            print(f"  - Total Gates : {total_gates}")
            print(f"  - Num Qubits  : {n_qubits}")
            print("="*60 + "\n")
        
        return recon_circ

    def get_original_layer_idx(self, ctx: CompilationContext, layer_range: tuple[int, int]) -> tuple[int, int]:
        """
        获取指定层在原始电路中的对应层号。

        Args:
            ctx (CompilationContext): 编译上下文。
            layer_range (tuple[int, int]): 指定的层范围。

        Returns:
            int: 原始电路中的层号。
        """
        left, right = layer_range
        if left < 0 or right < left or right >= len(ctx.multiq_layers):
            raise ValueError(
                f"invalid multiq layer range: {layer_range}, "
                f"multiq_layers={len(ctx.multiq_layers)}"
            )

        if left == 0:
            ori_left = 0
        else:
            ori_left = ctx.map_to_circuit_layer[left - 1] + 1

        if right == len(ctx.multiq_layers) - 1:
            ori_right = len(ctx.circuit_layers) - 1
        else:
            ori_right = ctx.map_to_circuit_layer[right]

        return (ori_left, ori_right)


@dataclass(frozen=True)
class NAVICheckpointEvaluationSpec:
    id: str
    name: str
    source: str
    policy: Optional[str] = None


@dataclass
class NAVICheckpointEvaluationResult:
    result_info: dict[str, dict[str, Any]]
    checkpoint_dir: Optional[Path] = None


def navi_checkpoint_enabled(global_config: dict[str, Any]) -> bool:
    cfg = global_config.get("navi_checkpoint")
    return isinstance(cfg, dict) and bool(cfg.get("enabled", False))


def parse_navi_checkpoint_evaluations(global_config: dict[str, Any]) -> list[NAVICheckpointEvaluationSpec]:
    cfg = global_config.get("navi_checkpoint")
    if not isinstance(cfg, dict) or not bool(cfg.get("enabled", False)):
        return []

    raw_items = cfg.get("evaluations", [])
    if raw_items is None:
        raw_items = []
    if not isinstance(raw_items, list):
        raise ValueError("navi_checkpoint.evaluations must be a list")

    specs: list[NAVICheckpointEvaluationSpec] = []
    seen_ids: set[str] = set()
    seen_names: set[str] = set()
    for idx, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            raise ValueError(f"navi_checkpoint.evaluations[{idx}] must be an object")
        eval_id = str(raw.get("id", "")).strip()
        if not eval_id:
            raise ValueError(f"navi_checkpoint.evaluations[{idx}].id is required")
        if eval_id in seen_ids:
            raise ValueError(f"duplicate NAVI checkpoint evaluation id: {eval_id}")
        name = str(raw.get("name", eval_id)).strip()
        if not name:
            raise ValueError(f"navi_checkpoint.evaluations[{idx}].name must not be empty")
        if name in seen_names:
            raise ValueError(f"duplicate NAVI checkpoint evaluation name: {name}")
        source = str(raw.get("source", "")).strip()
        if not source:
            raise ValueError(f"navi_checkpoint.evaluations[{idx}].source is required")
        policy = raw.get("policy")
        if policy is not None:
            policy = str(policy).strip() or None

        specs.append(NAVICheckpointEvaluationSpec(id=eval_id, name=name, source=source, policy=policy))
        seen_ids.add(eval_id)
        seen_names.add(name)
    return specs


def load_mapping_record_list_from_raw_json(path: Path) -> MappingRecordList:
    data = json.loads(path.read_text(encoding="utf-8"))
    record_list = MappingRecordList()
    for item in data.get("records", []):
        extra_info = item.get("extra_info")
        extra_info = MappingRecordList._deserialize_extra_info(extra_info)
        record = MappingRecord(
            layer_start=int(item.get("layer_start", -1)),
            layer_end=int(item.get("layer_end", -1)),
            partition=[[int(q) for q in group] for group in item.get("partition", [])],
            mapping_type=str(item.get("mapping_type", "")),
            extra_info=copy.deepcopy(extra_info) if extra_info is not None else None,
        )
        record_list.add_record(record)
    record_list.num_records = len(record_list.records)
    return record_list


def _parse_checkpoint_manifest_entry(entry: Any, source: str) -> tuple[str, Optional[float]]:
    if isinstance(entry, str):
        return entry, None
    if isinstance(entry, dict):
        rel_path = entry.get("file")
        if not rel_path:
            rel_path = entry.get("path")
        if not rel_path:
            raise ValueError(f"NAVI checkpoint manifest entry '{source}' must include a file")

        execution_time = entry.get("execution_time")
        if execution_time is None:
            return str(rel_path), None
        return str(rel_path), float(execution_time)

    raise ValueError(
        f"NAVI checkpoint manifest entry '{source}' must be a filename string or object, "
        f"got {type(entry)}"
    )


def evaluate_navi_checkpoints_from_config(
    global_config: dict[str, Any],
    checkpoint_dir: str | Path | None,
    network: Network,
    task_info: dict[str, Any],
    default_policy_name: Optional[str] = None,
    config: Optional[dict[str, Any]] = None,
) -> NAVICheckpointEvaluationResult:
    specs = parse_navi_checkpoint_evaluations(global_config)
    if not specs or checkpoint_dir is None:
        return NAVICheckpointEvaluationResult(result_info={}, checkpoint_dir=None)

    checkpoint_path = Path(checkpoint_dir)
    manifest_path = checkpoint_path / "manifest.json"
    if not manifest_path.exists():
        print(f"[INFO] NAVI checkpoint manifest not found; skipping evaluations: {manifest_path}")
        return NAVICheckpointEvaluationResult(result_info={}, checkpoint_dir=checkpoint_path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"NAVI checkpoint manifest must be a JSON object: {manifest_path}")

    result_info: dict[str, dict[str, Any]] = {}
    for spec in specs:
        manifest_entry = manifest.get(spec.source)
        if manifest_entry is None:
            available = ", ".join(sorted(str(key) for key in manifest.keys()))
            raise ValueError(f"NAVI checkpoint source '{spec.source}' not found in manifest. Available: [{available}]")
        rel_path, execution_time = _parse_checkpoint_manifest_entry(manifest_entry, spec.source)
        raw_path = checkpoint_path / str(rel_path)
        if not raw_path.exists():
            raise FileNotFoundError(f"NAVI checkpoint raw file not found: {raw_path}")

        record_list = load_mapping_record_list_from_raw_json(raw_path)
        pre_eval_time = float(execution_time) if execution_time is not None else 0.0
        print(
            f"[TIMING] [{spec.name}] pre-evaluate raw mapping record time: "
            f"{pre_eval_time:.6f} sec"
        )
        eval_start_time = time.time()
        eval_config = dict(config or {})
        eval_config["_compiler_id"] = "navi"
        evaluated = CompilerUtils.evaluate_raw_mapping_records(
            record_list,
            network,
            policy_name=spec.policy or default_policy_name,
            config=eval_config,
        )
        eval_time = time.time() - eval_start_time
        print(
            f"[TIMING] [{spec.name}] evaluate raw mapping record time: "
            f"{eval_time:.6f} sec"
        )
        if execution_time is not None:
            evaluated.update_total_costs(execution_time=execution_time + eval_time)
        result_info[spec.name] = {
            "F_eff": np.exp(evaluated.total_costs.total_fidelity_log_sum / task_info["Depth"]),
            **evaluated.total_costs.to_dict(),
        }

    return NAVICheckpointEvaluationResult(result_info=result_info, checkpoint_dir=checkpoint_path)
