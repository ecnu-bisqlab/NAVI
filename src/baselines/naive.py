from __future__ import annotations

import datetime
import time
from pathlib import Path
from typing import Any, Optional

from qiskit import QuantumCircuit

from ..compiler import Compiler, CompilerUtils, MappingRecord, MappingRecordList
from ..utils import Network


class Naive(Compiler):
    """Static contiguous placement with synthetic TeleGate execution."""

    compiler_id = "naive"

    @property
    def name(self) -> str:
        return "Naive"

    def _record_base_path(
        self,
        circuit_name: str,
        network: Network,
        config: dict[str, Any],
    ) -> str:
        configured = config.get("record_output_base")
        if configured:
            return str(Path(str(configured)))

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_folder = str(config.get("_output_folder", "./outputs"))
        return CompilerUtils.build_record_base_path(
            circuit_name,
            network,
            self.name,
            timestamp,
            output_folder=output_folder,
        )

    def compile(
        self,
        circuit: QuantumCircuit,
        network: Network,
        config: Optional[dict[str, Any]] = None,
    ) -> MappingRecordList:
        """Evaluate one full-circuit TeleGate record without graph partitioning."""
        cfg = dict(config or {})
        circuit_name = str(cfg.get("circuit_name", "circ"))
        print(f"Compiling with [{self.name}]...")
        start_time = time.time()

        partition = CompilerUtils.allocate_qubits(circuit.num_qubits, network)
        record = MappingRecord(
            layer_start=0,
            layer_end=circuit.depth() - 1,
            partition=partition,
            mapping_type="telegate",
        )
        records = MappingRecordList()
        records.add_record(record)
        records.num_records = 1

        circuit_layers = CompilerUtils.build_circuit_layers(circuit)
        records.ensure_record_ops(circuit, circuit_layers)
        base_path = self._record_base_path(circuit_name, network, cfg)
        records.save_records(f"{base_path}_raw.json", dump_type="raw")

        policy_name = cfg.get("evaluator_policy")
        records, _, _ = self._evaluate_raw_mapping_records_with_timing(
            records,
            network,
            policy_name=policy_name,
            config=cfg,
            start_time=start_time,
        )
        records.update_total_costs(execution_time=time.time() - start_time)
        records.save_records(f"{base_path}_evaluated.json", dump_type="evaluated")
        return records
