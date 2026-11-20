import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import networkx as nx

from src.baselines.wbcp import WBCP
from scripts.analyze_oee_iterations import (
    _write_outputs,
    canonicalize_partition,
    parse_args,
    run_cap_sweep,
    weighted_cut,
)
from scripts.summarize_oee_iteration_suite import (
    BENCHMARKS,
    collect_suite,
    write_suite,
)
from scripts.summarize_wbcp_segment_suite import (
    collect_suite as collect_wbcp_suite,
    write_suite as write_wbcp_suite,
)


class OEEIterationAnalysisTest(unittest.TestCase):
    def setUp(self):
        self.qig = nx.Graph()
        self.qig.add_nodes_from(range(4))
        self.qig.add_edge(0, 2, weight=5)
        self.qig.add_edge(1, 3, weight=5)
        self.initial_partition = [[0, 1], [2, 3]]
        self.network = SimpleNamespace(num_backends=2)

    def test_cap_sweep_finds_first_unchanged_partition(self):
        rows, convergence = run_cap_sweep(
            self.initial_partition,
            self.qig,
            self.network,
            target_cap=3,
            reference_cap=5,
        )

        rows_by_cap = {row["iteration_cap"]: row for row in rows}
        self.assertEqual(weighted_cut(self.qig, self.initial_partition), 10)
        self.assertEqual(rows_by_cap[0]["weighted_cut"], 10)
        self.assertEqual(rows_by_cap[1]["weighted_cut"], 0)
        self.assertEqual(convergence["first_unchanged_cap"], 2)
        self.assertEqual(convergence["accepted_iterations"], 1)
        self.assertEqual(convergence["target_weighted_cut"], 0)
        self.assertEqual(convergence["reference_weighted_cut"], 0)
        self.assertTrue(convergence["target_reference_match"])
        self.assertTrue(convergence["sufficient"])

    def test_partition_signature_ignores_order_within_each_qpu(self):
        self.assertEqual(
            canonicalize_partition([[1, 0], [3, 2]]),
            canonicalize_partition([[0, 1], [2, 3]]),
        )
        self.assertNotEqual(
            canonicalize_partition([[0, 2], [1, 3]]),
            canonicalize_partition([[0, 1], [2, 3]]),
        )

    def test_cli_accepts_every_supported_benchmark(self):
        benchmarks = (
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
        for benchmark in benchmarks:
            with self.subTest(benchmark=benchmark):
                self.assertEqual(
                    parse_args(["--benchmark", benchmark]).benchmark,
                    benchmark,
                )

    def test_wbcp_window_plan_preserves_default_and_supports_sweep(self):
        self.assertEqual(WBCP._resolve_window_plan(265), (30, 9))
        self.assertEqual(
            WBCP._resolve_window_plan(265, {"num_segments": 20}),
            (30, 9),
        )
        self.assertEqual(WBCP._resolve_window_plan(3571, {"num_segments": 5}), (500, 8))
        self.assertEqual(WBCP._resolve_window_plan(791, {"num_segments": 20}), (40, 20))
        with self.assertRaisesRegex(ValueError, "positive integer"):
            WBCP._resolve_window_plan(265, {"num_segments": 0})

    def test_wbcp_explicit_window_plans_stay_within_bounds(self):
        for depth in (1, 103, 265, 791, 3571, 20000):
            for requested_segments in (5, 10, 15, 20, 25, 30):
                with self.subTest(depth=depth, requested_segments=requested_segments):
                    window_length, actual_segments = WBCP._resolve_window_plan(
                        depth,
                        {"num_segments": requested_segments},
                    )
                    self.assertGreaterEqual(window_length, 30)
                    self.assertLessEqual(window_length, 500)
                    self.assertEqual(actual_segments, math.ceil(depth / window_length))

    def test_writes_csv_json_png_and_pdf(self):
        rows, convergence = run_cap_sweep(
            self.initial_partition,
            self.qig,
            self.network,
            target_cap=3,
            reference_cap=5,
        )
        context = {
            "benchmark": "Synthetic",
            "circuit_seed": 42,
            "requested_qubits": 4,
            "actual_qubits": 4,
            "qpu_layout": "2x2",
            "network_topology": "all_to_all",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            artifacts = _write_outputs(
                output_dir,
                rows,
                convergence,
                context,
            )

            with artifacts["csv"].open(encoding="utf-8", newline="") as handle:
                csv_rows = list(csv.DictReader(handle))
            summary = json.loads(artifacts["summary"].read_text(encoding="utf-8"))

            self.assertEqual(len(csv_rows), len(rows))
            self.assertEqual(csv_rows[0]["benchmark"], "Synthetic")
            self.assertTrue(summary["verdict"]["sufficient"])
            self.assertEqual(artifacts["png"].read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(artifacts["pdf"].read_bytes()[:4], b"%PDF")

    def test_collects_complete_benchmark_suite(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for index, benchmark in enumerate(BENCHMARKS):
                run_dir = root / benchmark.lower()
                run_dir.mkdir()
                (run_dir / "summary.json").write_text(
                    json.dumps(
                        {
                            "experiment": {
                                "benchmark": benchmark,
                                "requested_qubits": 200,
                                "actual_qubits": 200,
                                "qpu_layout": "4x50",
                                "circuit_seed": 42,
                            },
                            "convergence": {
                                "initial_weighted_cut": 100 + index,
                                "accepted_iterations": index,
                                "first_unchanged_cap": index + 1,
                                "target_cap": 50,
                                "target_weighted_cut": 50,
                                "reference_cap": 100,
                                "reference_weighted_cut": 50,
                                "target_reference_match": True,
                                "sufficient": True,
                            },
                        }
                    ),
                    encoding="utf-8",
                )

            rows = collect_suite(root, 200, "4x50", 42)
            artifacts = write_suite(root / "suite", rows)
            report = json.loads(artifacts["json"].read_text(encoding="utf-8"))

            self.assertEqual([row["benchmark"] for row in rows], list(BENCHMARKS))
            self.assertEqual(report["benchmark_count"], 12)
            self.assertEqual(report["maximum_accepted_iterations"], 11)
            self.assertTrue(report["all_sufficient"])

    def test_collects_wbcp_suite_and_preserves_quality_ties(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for benchmark in BENCHMARKS:
                run_dir = root / benchmark.lower()
                run_dir.mkdir()
                sweep_rows = [
                    {
                        "configuration": "legacy_default",
                        "requested_segments": None,
                        "actual_segments": 9,
                        "epr_pairs": 90,
                        "remote_effective_fidelity": 0.8,
                        "total_time_sec": 9.0,
                        "evaluation_mode": "communication",
                    }
                ]
                for segments in (5, 10, 15, 20, 25, 30):
                    epr_pairs = 100 + abs(segments - 10)
                    if benchmark == "VQE" and segments in (5, 10):
                        epr_pairs = 80
                    sweep_rows.append(
                        {
                            "configuration": f"target_{segments}",
                            "requested_segments": segments,
                            "actual_segments": segments,
                            "epr_pairs": epr_pairs,
                            "remote_effective_fidelity": 0.9,
                            "total_time_sec": float(segments),
                            "evaluation_mode": "communication",
                        }
                    )
                (run_dir / "summary.json").write_text(
                    json.dumps(
                        {
                            "experiment": {
                                "benchmark": benchmark,
                                "requested_qubits": 200,
                                "actual_qubits": 200,
                                "circuit_depth": 100,
                                "qpu_layout": "4x50",
                                "circuit_seed": 42,
                                "network_topology": "all_to_all",
                                "window_length_bounds": [30, 500],
                            },
                            "rows": sweep_rows,
                        }
                    ),
                    encoding="utf-8",
                )

            rows, matrix = collect_wbcp_suite(root, 200, "4x50", 42)
            artifacts = write_wbcp_suite(root / "suite", rows, matrix)
            report = json.loads(artifacts["json"].read_text(encoding="utf-8"))

            self.assertEqual([row["benchmark"] for row in rows], list(BENCHMARKS))
            self.assertEqual(rows[0]["best_requested_segments"], 10)
            self.assertEqual(rows[0]["overall_best_configuration"], "legacy_default")
            vqe = next(row for row in rows if row["benchmark"] == "VQE")
            self.assertEqual(vqe["quality_tied_requested_segments"], "5,10")
            self.assertEqual(vqe["best_requested_segments"], 5)
            self.assertEqual(report["benchmark_count"], 12)
            self.assertEqual(artifacts["png"].read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(artifacts["pdf"].read_bytes()[:4], b"%PDF")


if __name__ == "__main__":
    unittest.main()
