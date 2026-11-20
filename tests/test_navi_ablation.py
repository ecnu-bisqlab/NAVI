import copy
import csv
import io
import hashlib
import json
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from main import _ordered_result_info
from src.compiler import CompilerUtils, ExecCosts, MappingRecord, MappingRecordList
from src.navi import (
    CompilationContext,
    InsufficientAtomicZeroCutSubcircuitsError,
    NAVI,
)
from src.navi.navi_compiler import evaluate_navi_checkpoints_from_config
from src.navi.partition_assigner import (
    DirectPartitionAssigner,
    GlobalMaxMatchPartitionAssigner,
    MaxMatchPartitionAssigner,
    PartitionAssignerFactory,
)
from src.utils.network import Network
from src.utils.qcircuits import iqp_circuit, mcmt_circuit, pqc
from scripts.analyze_commopt_parameters import build_dataset


class NAVIAblationTest(unittest.TestCase):
    def test_parameter_analysis_rejects_too_few_atomic_zero_cut_subcircuits(self):
        navi = NAVI()
        output = io.StringIO()
        ranges = [(index, index) for index in range(4)]

        with (
            patch(
                "src.navi.navi_compiler.PartitionerFactory.create_partitioner",
                return_value=SimpleNamespace(),
            ),
            patch.object(navi, "_step_remove_single_qubit_gates"),
            patch.object(navi, "_step_build_partition_table", return_value=[]),
            patch.object(navi, "_step_build_greedy_slicing", return_value=ranges),
            redirect_stdout(output),
            self.assertRaises(InsufficientAtomicZeroCutSubcircuitsError) as raised,
        ):
            navi._build_zero_cut_context(
                SimpleNamespace(),
                SimpleNamespace(),
                {
                    "benchmark": "MCMT",
                    "min_atomic_zero_cut_subcircuits": 15,
                },
            )

        self.assertEqual(raised.exception.benchmark, "MCMT")
        self.assertEqual(raised.exception.actual, 4)
        self.assertEqual(raised.exception.minimum, 15)
        marker = next(
            line
            for line in output.getvalue().splitlines()
            if line.startswith("[COMMOPT_INELIGIBLE] ")
        )
        payload = json.loads(marker.removeprefix("[COMMOPT_INELIGIBLE] "))
        self.assertEqual(payload["atomic_zero_cut_subcircuit_count"], 4)
        self.assertEqual(payload["minimum_required"], 15)

    def test_direct_assigner_skips_max_match(self):
        candidates = [
            [[ [0, 1], [2, 3] ]],
            [[ [2, 3], [0, 1] ]],
        ]

        direct = DirectPartitionAssigner().assign(copy.deepcopy(candidates))["partition_plan"]
        matched = MaxMatchPartitionAssigner().assign(copy.deepcopy(candidates))["partition_plan"]

        self.assertEqual(direct[1], [[2, 3], [0, 1]])
        self.assertEqual(matched[0], matched[1])

    def test_max_match_applies_hungarian_pairs_in_the_correct_direction(self):
        candidates = [
            [[ [0, 1], [2, 3], [4, 5] ]],
            [[ [2, 3], [4, 5], [0, 1] ]],
            [[ [4, 5], [0, 1], [2, 3] ]],
        ]

        result = MaxMatchPartitionAssigner().assign(copy.deepcopy(candidates))

        self.assertEqual(result["costs"]["num_moves"], 0)
        self.assertEqual(result["partition_plan"], [
            [[0, 1], [2, 3], [4, 5]],
            [[0, 1], [2, 3], [4, 5]],
            [[0, 1], [2, 3], [4, 5]],
        ])

    def test_max_match_and_global_max_match_have_equal_movement_cost(self):
        rng = random.Random(42)
        candidates = []
        for _ in range(6):
            qubits = list(range(20))
            rng.shuffle(qubits)
            candidates.append([[qubits[index::4] for index in range(4)]])

        local_result = MaxMatchPartitionAssigner().assign(copy.deepcopy(candidates))
        global_result = GlobalMaxMatchPartitionAssigner().assign(copy.deepcopy(candidates))

        self.assertEqual(
            local_result["costs"]["num_moves"],
            global_result["costs"]["num_moves"],
        )

    def test_partition_assigner_factory_rejects_disabled_global_max_match(self):
        with self.assertRaisesRegex(ValueError, "global_max_match"):
            PartitionAssignerFactory.create_assigner("global_max_match")

    def test_ablation_uses_direct_checkpoints_and_max_match_for_full_navi(self):
        candidates = [
            [[ [0, 1], [2, 3] ]],
            [[ [2, 3], [0, 1] ]],
        ]
        base_ctx = CompilationContext(
            circuit=SimpleNamespace(),
            network=SimpleNamespace(net_type="mesh_grid"),
            config={
                "ablation_mode": True,
                "navi_checkpoint_enabled": True,
                "partition_assigner": "max_match",
                "multi_record_mapper": "boundeddp_neighbor",
            },
            partition_candidates=candidates,
        )
        navi = NAVI()
        saved = {}
        hybrid_plans = []

        def prepare(ctx, config):
            prepared = copy.deepcopy(ctx)
            prepared.config = copy.deepcopy(config)
            return prepared

        def teledata(ctx):
            return MappingRecordList(records=[
                MappingRecord(partition=ctx.partition_plan[1], mapping_type="teledata")
            ])

        def hybrid(ctx):
            hybrid_plans.append(copy.deepcopy(ctx.partition_plan))
            return MappingRecordList(records=[
                MappingRecord(partition=ctx.partition_plan[1], mapping_type="cat")
            ])

        def save(ctx, key, filename, records, execution_time=None):
            saved[key] = (copy.deepcopy(ctx.partition_plan), records, execution_time)

        def map_records(ctx, records, mapper_id, shared_prefix_time=0.0, records_time=0.0):
            self.assertEqual(mapper_id, "single_record_greedy")
            self.assertEqual(ctx.partition_plan[0], ctx.partition_plan[1])
            return records

        with (
            patch.object(navi, "_build_zero_cut_context", return_value=(base_ctx, 1.0)),
            patch.object(navi, "_prepare_ctx_from_shared_prefix", side_effect=prepare),
            patch.object(navi, "_step_construct_teledata_only_records", side_effect=teledata),
            patch.object(navi, "_step_construct_hybrid_records_beam", side_effect=hybrid),
            patch.object(navi, "_save_checkpoint_records", side_effect=save),
            patch.object(navi, "_map_records_with_mapper", side_effect=map_records),
            patch.object(MappingRecordList, "ensure_record_ops", return_value=None),
        ):
            result = navi._compile_ablation(base_ctx.circuit, base_ctx.network, base_ctx.config)

        self.assertEqual(set(saved), {"NAVI-Td", "NAVI-Hy"})
        self.assertEqual(saved["NAVI-Td"][0][1], [[2, 3], [0, 1]])
        self.assertEqual(saved["NAVI-Hy"][0][1], [[2, 3], [0, 1]])
        self.assertEqual(hybrid_plans[0][1], [[2, 3], [0, 1]])
        self.assertEqual(hybrid_plans[1][0], hybrid_plans[1][1])
        self.assertEqual(result.records[0].mapping_type, "cat")

    def test_checkpoint_execution_time_includes_its_own_evaluation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir)
            (checkpoint_dir / "manifest.json").write_text(
                json.dumps({
                    "NAVI-Td": {
                        "file": "NAVI-Td_raw.json",
                        "execution_time": 5.0,
                    }
                }),
                encoding="utf-8",
            )
            (checkpoint_dir / "NAVI-Td_raw.json").write_text(
                json.dumps({
                    "records": [{
                        "layer_start": 0,
                        "layer_end": 0,
                        "partition": [[0], [1]],
                        "mapping_type": "teledata",
                    }]
                }),
                encoding="utf-8",
            )
            evaluated = MappingRecordList(total_costs=ExecCosts())
            config = {
                "navi_checkpoint": {
                    "enabled": True,
                    "evaluations": [{
                        "id": "navi_td",
                        "name": "NAVI-Td",
                        "source": "NAVI-Td",
                    }],
                }
            }

            with (
                patch(
                    "src.navi.navi_compiler.CompilerUtils.evaluate_raw_mapping_records",
                    return_value=evaluated,
                ),
                patch("src.navi.navi_compiler.time.time", side_effect=[10.0, 12.0]),
            ):
                result = evaluate_navi_checkpoints_from_config(
                    config,
                    checkpoint_dir,
                    SimpleNamespace(),
                    {"Depth": 1},
                )

        self.assertEqual(result.result_info["NAVI-Td"]["execution_time"], 7.0)

    def test_telegate_range_cache_reuses_interval_across_previous_partitions(self):
        class RecordingPartitioner:
            def __init__(self):
                self.partitions = []

            def partition(self, circuit, network, config):
                self.partitions.append(copy.deepcopy(config["partition"]))
                return MappingRecord(
                    layer_start=config["layer_start"],
                    layer_end=config["layer_end"],
                    partition=config["partition"],
                    mapping_type="cat",
                )

        partitioner = RecordingPartitioner()
        ctx = CompilationContext(
            circuit=SimpleNamespace(),
            network=SimpleNamespace(),
            config={"telegate_cache": True, "telegate_cache_mode": "range"},
            telegate_partitioner=partitioner,
        )
        ctx.runtime_hybrid_beam_width = 2
        first_partition = [[0, 1], [2, 3]]
        second_partition = [[2, 3], [0, 1]]
        navi = NAVI()

        with (
            patch.object(navi, "_get_ori_subc", return_value=SimpleNamespace()),
            patch.object(navi, "_set_simple_telegate_ops_for_full_circuit_record"),
        ):
            navi._try_generate_telegate(
                ctx, 0, 5, prev_partition=first_partition
            )
            navi._try_generate_telegate(
                ctx, 0, 5, prev_partition=first_partition
            )
            navi._try_generate_telegate(
                ctx, 0, 5, prev_partition=second_partition
            )

        self.assertEqual(partitioner.partitions, [first_partition])
        self.assertEqual(ctx.telegate_cache_relaxed_hits, 2)
        self.assertEqual(len(ctx.telegate_cache), 1)

    def test_hybrid_uses_previous_partition_and_none_at_circuit_start(self):
        partitions = [
            [[0, 2], [1, 3]],
            [[0, 3], [1, 2]],
            [[0, 3], [1, 2]],
        ]
        ctx = CompilationContext(
            circuit=SimpleNamespace(num_qubits=4),
            network=SimpleNamespace(backend_sizes=[2, 2], num_backends=2),
            config={
                "hybrid_beam_width": 1,
                "hybrid_max_merge_span": 2,
                "hybrid_merge_probe_budget": 2,
                "hybrid_pos_probe_budget": 3,
            },
            subc_ranges=[(0, 0), (1, 1), (2, 2)],
            partition_plan=partitions,
            partition_assigner=DirectPartitionAssigner(),
        )
        navi = NAVI()
        telegate_calls = []
        transition_calls = []

        def generate(ctx, start, end, prev_partition=None):
            telegate_calls.append((start, end, copy.deepcopy(prev_partition)))
            record = MappingRecord(
                layer_start=start,
                layer_end=end,
                partition=[[0, 1], [2, 3]],
                mapping_type="cat",
                costs=ExecCosts(epairs=100),
                extra_info={"ops": SimpleNamespace()},
            )
            return MappingRecordList(
                total_costs=ExecCosts(epairs=100, cat_ents=1),
                records=[record],
            )

        def transition(previous, current, network):
            transition_calls.append(
                (copy.deepcopy(previous), copy.deepcopy(current))
            )
            return ExecCosts(), None

        with (
            patch.object(navi, "get_original_layer_idx", side_effect=lambda ctx, value: value),
            patch.object(navi, "_try_generate_telegate", side_effect=generate),
            patch(
                "src.navi.navi_compiler.CompilerUtils.evaluate_teledata",
                side_effect=transition,
            ),
            patch(
                "src.navi.navi_compiler.CompilerUtils.evaluate_teledata_with_local",
                return_value=(ExecCosts(), {}, {}),
            ),
            patch(
                "src.navi.navi_compiler.CompilerUtils.evaluate_local_and_telegate_with_cat",
                return_value=(ExecCosts(), {}, {}),
            ),
            patch(
                "src.navi.navi_compiler.CompilerUtils.get_subcircuit_by_level",
                return_value=SimpleNamespace(),
            ),
        ):
            navi._step_construct_hybrid_records_beam(ctx)

        calls_by_range = {
            (start, end): previous
            for start, end, previous in telegate_calls
        }
        self.assertIsNone(calls_by_range[(0, 1)])
        self.assertIsNone(calls_by_range[(0, 2)])
        self.assertEqual(calls_by_range[(1, 2)], partitions[0])
        self.assertIn((partitions[0], [[0, 1], [2, 3]]), transition_calls)

    def test_disabled_full_span_safeguard_respects_lmax_but_allows_natural_full_span(self):
        partitions = [
            [[0, 1], [2, 3]],
            [[0, 1], [2, 3]],
            [[0, 1], [2, 3]],
        ]

        def run_with_span(max_merge_span):
            ctx = CompilationContext(
                circuit=SimpleNamespace(num_qubits=4),
                network=SimpleNamespace(backend_sizes=[2, 2], num_backends=2),
                config={
                    "hybrid_beam_width": 1,
                    "hybrid_max_merge_span": max_merge_span,
                    "hybrid_merge_probe_budget": 3,
                    "hybrid_pos_probe_budget": 3,
                    "hybrid_full_span_safeguard": False,
                },
                subc_ranges=[(0, 0), (1, 1), (2, 2)],
                partition_plan=partitions,
                partition_assigner=DirectPartitionAssigner(),
            )
            navi = NAVI()
            telegate_calls = []

            def generate(ctx, start, end, prev_partition=None):
                telegate_calls.append(
                    (start, end, copy.deepcopy(prev_partition))
                )
                record = MappingRecord(
                    layer_start=start,
                    layer_end=end,
                    partition=[[0, 1], [2, 3]],
                    mapping_type="cat",
                    costs=ExecCosts(),
                    extra_info={"ops": SimpleNamespace()},
                )
                return MappingRecordList(
                    total_costs=ExecCosts(cat_ents=1),
                    records=[record],
                )

            with (
                patch.object(
                    navi,
                    "get_original_layer_idx",
                    side_effect=lambda ctx, value: value,
                ),
                patch.object(navi, "_try_generate_telegate", side_effect=generate),
                patch(
                    "src.navi.navi_compiler.CompilerUtils.evaluate_teledata",
                    return_value=(ExecCosts(), None),
                ),
                patch(
                    "src.navi.navi_compiler.CompilerUtils.evaluate_teledata_with_local",
                    return_value=(ExecCosts(), {}, {}),
                ),
                patch(
                    "src.navi.navi_compiler.CompilerUtils.evaluate_local_and_telegate_with_cat",
                    return_value=(ExecCosts(), {}, {}),
                ),
                patch(
                    "src.navi.navi_compiler.CompilerUtils.get_subcircuit_by_level",
                    return_value=SimpleNamespace(),
                ),
            ):
                result = navi._step_construct_hybrid_records_beam(ctx)
            return telegate_calls, result, ctx

        limited_calls, limited_result, limited_ctx = run_with_span(2)
        self.assertFalse(any(call[:2] == (0, 2) for call in limited_calls))
        self.assertGreater(len(limited_result.records), 1)
        self.assertFalse(limited_ctx.commopt_full_span_won)

        natural_calls, natural_result, natural_ctx = run_with_span(3)
        full_span_calls = [call for call in natural_calls if call[:2] == (0, 2)]
        self.assertEqual(full_span_calls, [(0, 2, None)])
        self.assertEqual(len(natural_result.records), 1)
        self.assertTrue(natural_ctx.commopt_full_span_won)

    def test_network_seed_is_repeatable_and_does_not_touch_global_rng(self):
        backends = [
            SimpleNamespace(
                num_qubits=52,
                basis_gates=["rz", "cx"],
                two_qubit_gates=["cx"],
                name=f"backend-{idx}",
            )
            for idx in range(4)
        ]
        config = {
            "type": "mesh_grid",
            "fidelity_range": [0.90, 0.93],
            "comm_slot_reserve": 2,
            "seed": 42,
        }

        random.seed(1234)
        expected_next = random.random()
        random.seed(1234)
        first = Network(config, backends, build_all_to_all_copy=False)
        actual_next = random.random()
        second = Network(config, backends, build_all_to_all_copy=False)

        self.assertEqual(first.network_coupling, second.network_coupling)
        self.assertEqual(actual_next, expected_next)

    def test_csv_order_is_td_hy_full_navi(self):
        ordered = _ordered_result_info({
            "NAVI": {},
            "NAVI-Hy": {},
            "NAVI-Td": {},
        })
        self.assertEqual(list(ordered), ["NAVI-Td", "NAVI-Hy", "NAVI"])

    def test_standard_path_emits_commopt_analysis_record_after_safeguard(self):
        teledata = MappingRecordList(
            total_costs=ExecCosts(epairs=7),
            records=[MappingRecord(mapping_type="teledata")],
        )
        hybrid = MappingRecordList(
            total_costs=ExecCosts(epairs=9),
            records=[MappingRecord(mapping_type="cat"), MappingRecord(mapping_type="teledata")],
        )
        ctx = CompilationContext(
            circuit=SimpleNamespace(),
            network=SimpleNamespace(),
            config={
                "benchmark": "Random",
                "circuit_seed": 42,
                "partition_assigner": "max_match",
                "hybrid_beam_width": 2,
                "hybrid_max_merge_span": 15,
                "hybrid_merge_probe_budget": 15,
            },
            subc_ranges=[(0, 0), (1, 1), (2, 2)],
            partition_assigner=MaxMatchPartitionAssigner(),
        )
        navi = NAVI()
        output = io.StringIO()

        with (
            patch.object(navi, "_step_construct_teledata_only_records", return_value=teledata),
            patch.object(navi, "_step_construct_hybrid_records_beam", return_value=hybrid),
            patch.object(navi, "_save_checkpoint_records"),
            patch("src.navi.navi_compiler.time.perf_counter", side_effect=[10.0, 12.5]),
            redirect_stdout(output),
        ):
            selected, _ = navi._construct_records_from_ctx(ctx)

        line = next(
            line for line in output.getvalue().splitlines()
            if line.startswith("[COMMOPT_ANALYSIS] ")
        )
        analysis = json.loads(line.removeprefix("[COMMOPT_ANALYSIS] "))
        self.assertIs(selected, teledata)
        self.assertEqual(analysis["K"], 3)
        self.assertEqual(analysis["W"], 2)
        self.assertEqual(analysis["L_max"], 15)
        self.assertEqual(analysis["effective_local_Lmax"], 3)
        self.assertEqual(analysis["circuit_seed"], 42)
        self.assertEqual(analysis["partition_assigner_name"], "Max-Match Partition Assigner")
        self.assertEqual(analysis["teledata_epairs_internal"], 7.0)
        self.assertEqual(analysis["hybrid_epairs_internal"], 9.0)
        self.assertEqual(analysis["selected_epairs_internal"], 7.0)
        self.assertEqual(analysis["selected_source"], "teledata")
        self.assertEqual(analysis["commopt_time_sec"], 2.5)

    def test_record_output_base_path_accepts_experiment_directory(self):
        network = SimpleNamespace(name="net[20, 20, 20, 20]", net_type="all_to_all")
        path = CompilerUtils.build_record_base_path(
            "Random80",
            network,
            "NAVI",
            "20260813_120000",
            output_folder="/tmp/commopt/W1_L10",
        )
        self.assertEqual(
            path,
            "/tmp/commopt/W1_L10/Random80/Random80_net[20, 20, 20, 20]_all_to_all_NAVI_20260813_120000",
        )

    def test_randomized_benchmarks_are_repeatable_with_local_seed(self):
        factories = (
            lambda seed: mcmt_circuit(20, gate_width=5, gate_count=3, seed=seed).decompose(reps=10),
            lambda seed: iqp_circuit(8, seed=seed),
            lambda seed: pqc(8, connections_per_qubit=2, seed=seed),
        )
        for factory in factories:
            with self.subTest(factory=factory):
                self.assertEqual(factory(42), factory(42))
                self.assertNotEqual(factory(42), factory(43))

    def test_randomized_benchmarks_default_to_fixed_seed(self):
        factories = (
            lambda: mcmt_circuit(20, gate_width=5, gate_count=3).decompose(reps=10),
            lambda: iqp_circuit(8),
            lambda: pqc(8, connections_per_qubit=2),
        )
        for factory in factories:
            with self.subTest(factory=factory):
                self.assertEqual(factory(), factory())

    def test_randomized_benchmarks_do_not_touch_global_rngs(self):
        random.seed(1234)
        expected_python = random.random()
        np.random.seed(1234)
        expected_numpy = float(np.random.random())

        random.seed(1234)
        np.random.seed(1234)
        mcmt_circuit(20, gate_width=5, gate_count=2, seed=42)
        iqp_circuit(8, seed=42)
        pqc(8, connections_per_qubit=2, seed=42)

        self.assertEqual(random.random(), expected_python)
        self.assertEqual(float(np.random.random()), expected_numpy)

    def test_multibenchmark_dataset_preserves_replicates_and_final_records(self):
        def write_run(
            run_dir,
            experiment_id,
            started_at,
            width,
            span,
            *,
            qubit_count=80,
            capacity=20,
            requested_qubit_count=None,
            expected_combinations=None,
        ):
            run_dir.mkdir(parents=True)
            config = {
                "compiler_config": {"navi": {
                    "partition_assigner": "max_match",
                    "hybrid_beam_width": width,
                    "hybrid_max_merge_span": span,
                    "hybrid_merge_probe_budget": 15,
                    "hybrid_full_span_safeguard": True,
                }},
                "network_config": {"seed": 42},
            }
            config_text = json.dumps(config)
            (run_dir / "config.json").write_text(config_text, encoding="utf-8")
            config_sha = hashlib.sha256(config_text.encode()).hexdigest()
            metadata = {
                "run_id": f"{experiment_id}/Random/W{width}_L{span}",
                "started_at": started_at,
                "ended_at": started_at,
                "exit_status": 0,
                "git_commit": "test",
                "git_dirty": False,
                "config_sha256": config_sha,
                "benchmark": "Random",
                "circuit_seed": 42,
                "backend": "ibm_marrakesh",
                "calibration_date": "2026-03-01",
            }
            if requested_qubit_count is not None:
                metadata["requested_qubit_count"] = requested_qubit_count
            if expected_combinations is not None:
                metadata["expected_parameter_combinations"] = expected_combinations
            if experiment_id != "legacy_random":
                metadata["experiment_id"] = experiment_id
            (run_dir / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            analysis = {
                "benchmark": "Random",
                "circuit_seed": 42,
                "K": 12,
                "W": width,
                "L_max": span,
                "effective_local_Lmax": min(12, span),
                "partition_assigner": "max_match",
                "partition_assigner_name": "Max-Match Partition Assigner",
                "merge_probe_budget": 15,
                "pos_probe_budget": 12,
                "teledata_epairs_internal": 10,
                "hybrid_epairs_internal": 8,
                "selected_epairs_internal": 8,
                "selected_source": "hybrid",
                "teledata_record_count": 4,
                "hybrid_record_count": 1,
                "selected_record_count": 1,
                "commopt_time_sec": 2.0,
                "telegate_total_calls": 3,
                "telegate_cache_hits": 0,
                "telegate_cache_relaxed_hits": 0,
                "telegate_cache_misses": 3,
                "full_span_safeguard_enabled": True,
                "full_span_candidate_won": True,
            }
            (run_dir / "run.log").write_text(
                "[COMMOPT_ANALYSIS] " + json.dumps(analysis) + "\n",
                encoding="utf-8",
            )
            (run_dir / "results.csv").write_text(
                "Circuit,#Qubits,#Gates,#2Q Gates,Depth,#QPUs,QPUs,net_type,Metrics,NAVI\n"
                f"Random,{qubit_count},100,50,20,4,\"[{capacity}, {capacity}, {capacity}, {capacity}]\",all_to_all,EPR Pair Consumption,8\n"
                f"Random,{qubit_count},100,50,20,4,\"[{capacity}, {capacity}, {capacity}, {capacity}]\",all_to_all,Execution Time,3\n",
                encoding="utf-8",
            )
            evaluated = {
                "num_records": 1,
                "total_costs": {"epairs": 8, "execution_time": 3},
                "records": [{
                    "layer_start": 0,
                    "layer_end": 19,
                    "mapping_type": "cat",
                    "costs": {"epairs": 8, "cat_ents": 2},
                }],
            }
            (run_dir / "Random80_NAVI_test_evaluated.json").write_text(
                json.dumps(evaluated), encoding="utf-8"
            )
            (run_dir / "resource_usage.txt").write_text(
                "process_wall_time_sec=4\npeak_rss_kb=1000\nprocess_exit_status=0\n",
                encoding="utf-8",
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            legacy = root / "commopt_parameter_analysis"
            repeated = root / "navi_param_runs" / "repeat_02" / "Random"
            partial_large = root / "navi_param_runs" / "large_4x50" / "Random"
            custom_subset = root / "navi_param_runs" / "custom_subset" / "Random"
            for width in (1, 2, 4):
                for span in (5, 10, 15):
                    write_run(legacy / f"W{width}_L{span}", "legacy_random", "2026-01-01T00:00:00", width, span)
                    write_run(repeated / f"W{width}_L{span}", "repeat_02", "2026-01-02T00:00:00", width, span)
            for span in (5, 10):
                write_run(
                    partial_large / f"W1_L{span}",
                    "large_4x50",
                    "2026-01-03T00:00:00",
                    1,
                    span,
                    qubit_count=200,
                    capacity=50,
                    requested_qubit_count=200,
                )
            expected_subset = [{"W": 2, "L_max": 5}, {"W": 4, "L_max": 5}]
            for width in (2, 4):
                write_run(
                    custom_subset / f"W{width}_L5",
                    "custom_subset",
                    "2026-01-04T00:00:00",
                    width,
                    5,
                    expected_combinations=expected_subset,
                )
            active_run = partial_large / "W1_L15"
            active_run.mkdir(parents=True)
            (active_run / "run_metadata.json").write_text(
                json.dumps({
                    "experiment_id": "large_4x50",
                    "run_id": "large_4x50/Random/W1_L15",
                    "started_at": "2026-01-03T00:00:01",
                }),
                encoding="utf-8",
            )
            output_csv = root / "outputs" / "navi_param.csv"
            validation = build_dataset([legacy, root / "navi_param_runs"], output_csv, root)
            with output_csv.open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertTrue(validation["valid"])
        self.assertFalse(validation["all_groups_complete"])
        self.assertEqual(len(rows), 22)
        defaults = [row for row in rows if row["W"] == "1" and row["L_max"] == "10"]
        self.assertEqual({row["replicate_id"] for row in defaults}, {"1", "2"})
        self.assertTrue(all(row["navi_final_record_count"] == "1" for row in rows))
        self.assertTrue(all(row["navi_final_cat_record_count"] == "1" for row in rows))
        self.assertTrue(all(row["full_span_safeguard_enabled"] == "True" for row in rows))
        full_grid_rows = [row for row in rows if row["experiment_id"] != "custom_subset"]
        self.assertTrue(all(row["hybrid_epairs_relative_default"] == "1.0" for row in full_grid_rows))
        old_rows = [row for row in rows if row["experiment_id"] != "large_4x50"]
        self.assertTrue(all(row["qpu_layout"] == "4x20" for row in old_rows))
        self.assertTrue(all(row["requested_qubit_count"] == "80" for row in old_rows))
        partial_rows = [row for row in rows if row["experiment_id"] == "large_4x50"]
        self.assertEqual(len(partial_rows), 2)
        self.assertTrue(all(row["is_valid"] == "True" for row in partial_rows))
        self.assertTrue(all(row["group_complete"] == "False" for row in partial_rows))
        self.assertTrue(all("missing parameter combinations" in row["group_validation_errors"] for row in partial_rows))
        self.assertTrue(all(row["qpu_layout"] == "4x50" for row in partial_rows))
        self.assertTrue(all(row["requested_qubit_count"] == "200" for row in partial_rows))
        subset_rows = [row for row in rows if row["experiment_id"] == "custom_subset"]
        self.assertEqual(len(subset_rows), 2)
        self.assertTrue(all(row["group_complete"] == "True" for row in subset_rows))


if __name__ == "__main__":
    unittest.main()
