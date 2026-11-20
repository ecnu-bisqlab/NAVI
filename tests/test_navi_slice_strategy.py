import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts.run_slice_strategy_comparison import validate_paired_configs
from src.navi import CompilationContext, NAVI


def partition_table_from_right_endpoints(right_endpoints):
    size = len(right_endpoints)
    table = [[[] for _ in range(size)] for _ in range(size)]
    for left, right_endpoint in enumerate(right_endpoints):
        for right in range(left, right_endpoint + 1):
            table[left][right] = [[[(left, right)]]]
    return table


def monotone_right_endpoint_sequences(size, prefix=()):
    index = len(prefix)
    if index == size:
        yield prefix
        return

    lower = max(index, prefix[-1] if prefix else index)
    for right_endpoint in range(lower, size):
        yield from monotone_right_endpoint_sequences(
            size,
            prefix + (right_endpoint,),
        )


class NAVISliceStrategyTest(unittest.TestCase):
    def setUp(self):
        self.navi = NAVI()

    @staticmethod
    def context(table, strategy="dp"):
        return CompilationContext(
            circuit=SimpleNamespace(),
            network=SimpleNamespace(),
            config={"slice_strategy": strategy},
            P_table=table,
        )

    def test_greedy_empty_partition_table(self):
        ctx = self.context([], "greedy_table")
        self.assertEqual(self.navi._step_build_greedy_slicing(ctx), [])

    def test_greedy_selects_whole_feasible_range(self):
        table = partition_table_from_right_endpoints((3, 3, 3, 3))
        ctx = self.context(table, "greedy_table")
        self.assertEqual(self.navi._step_build_greedy_slicing(ctx), [(0, 3)])

    def test_greedy_selects_singletons_when_only_diagonal_is_feasible(self):
        table = partition_table_from_right_endpoints((0, 1, 2, 3))
        ctx = self.context(table, "greedy_table")
        self.assertEqual(
            self.navi._step_build_greedy_slicing(ctx),
            [(0, 0), (1, 1), (2, 2), (3, 3)],
        )

    def test_dp_and_greedy_can_choose_different_optimal_boundaries(self):
        table = partition_table_from_right_endpoints((1, 3, 3, 3))
        dp_ctx = self.context(table, "dp")
        greedy_ctx = self.context(table, "greedy_table")

        dp_ranges = self.navi._step_build_slicing_table(dp_ctx)
        greedy_ranges = self.navi._step_build_greedy_slicing(greedy_ctx)

        self.assertEqual(dp_ranges, [(0, 0), (1, 3)])
        self.assertEqual(greedy_ranges, [(0, 1), (2, 3)])
        self.assertEqual(len(dp_ranges), len(greedy_ranges))

    def test_greedy_ranges_are_contiguous_feasible_and_right_maximal(self):
        table = partition_table_from_right_endpoints((2, 3, 4, 5, 5, 5))
        ctx = self.context(table, "greedy_table")

        ranges = self.navi._step_build_greedy_slicing(ctx)

        self.assertEqual(ranges, [(0, 2), (3, 5)])
        next_left = 0
        for left, right in ranges:
            self.assertEqual(left, next_left)
            self.assertTrue(table[left][right])
            self.assertTrue(right == len(table) - 1 or not table[left][right + 1])
            next_left = right + 1
        self.assertEqual(next_left, len(table))

    def test_greedy_rejects_an_infeasible_single_layer(self):
        table = partition_table_from_right_endpoints((0, 1, 2))
        table[1][1] = []
        ctx = self.context(table, "greedy_table")
        with self.assertRaisesRegex(RuntimeError, r"P\[1\]\[1\] is empty"):
            self.navi._step_build_greedy_slicing(ctx)

    def test_all_small_monotone_tables_match_dp_segment_count(self):
        checked = 0
        for size in range(1, 9):
            for right_endpoints in monotone_right_endpoint_sequences(size):
                table = partition_table_from_right_endpoints(right_endpoints)
                dp_ranges = self.navi._step_build_slicing_table(
                    self.context(table, "dp")
                )
                greedy_ranges = self.navi._step_build_greedy_slicing(
                    self.context(table, "greedy_table")
                )
                self.assertEqual(
                    len(dp_ranges),
                    len(greedy_ranges),
                    msg=f"right_endpoints={right_endpoints}",
                )
                checked += 1
        self.assertGreater(checked, 1000)

    def test_dispatcher_defaults_to_greedy_and_explicit_greedy_is_identical(self):
        table = partition_table_from_right_endpoints((0,))
        for config in ({}, {"slice_strategy": "greedy_table"}):
            ctx = CompilationContext(
                circuit=SimpleNamespace(),
                network=SimpleNamespace(),
                config=config,
                P_table=table,
            )
            with (
                patch.object(
                    self.navi,
                    "_step_build_slicing_table",
                    side_effect=AssertionError("DP must not run"),
                ),
                patch.object(
                    self.navi,
                    "_step_build_greedy_slicing",
                    return_value=[(0, 0)],
                ) as greedy,
            ):
                self.assertEqual(self.navi._step_select_slicing(ctx), [(0, 0)])
            greedy.assert_called_once_with(ctx)
            self.assertEqual(ctx.slice_strategy, "greedy_table")
            self.assertGreaterEqual(ctx.slice_planning_time_sec, 0.0)

    def test_dispatcher_selects_greedy_without_building_s_or_t(self):
        table = partition_table_from_right_endpoints((1, 1))
        ctx = self.context(table, "greedy_table")
        with patch.object(
            self.navi,
            "_step_build_slicing_table",
            side_effect=AssertionError("DP must not run"),
        ):
            self.assertEqual(self.navi._step_select_slicing(ctx), [(0, 1)])
        self.assertEqual(ctx.S_table, [])
        self.assertEqual(ctx.T_table, [])
        self.assertEqual(ctx.slice_strategy, "greedy_table")

    def test_unknown_strategy_is_rejected_before_partition_table_work(self):
        with (
            patch(
                "src.navi.navi_compiler.PartitionerFactory.create_partitioner",
                side_effect=AssertionError("partitioner must not be created"),
            ),
            self.assertRaisesRegex(ValueError, "dp, greedy_table"),
        ):
            self.navi._build_zero_cut_context(
                SimpleNamespace(),
                SimpleNamespace(),
                {"slice_strategy": "unknown"},
            )

    def test_both_strategies_extract_candidates_from_their_selected_ranges(self):
        table = partition_table_from_right_endpoints((1, 3, 3, 3))
        expected = {
            "dp": [(0, 0), (1, 3)],
            "greedy_table": [(0, 1), (2, 3)],
        }

        for strategy, expected_ranges in expected.items():
            with (
                patch(
                    "src.navi.navi_compiler.PartitionerFactory.create_partitioner",
                    return_value=SimpleNamespace(),
                ),
                patch.object(self.navi, "_step_remove_single_qubit_gates"),
                patch.object(
                    self.navi,
                    "_step_build_partition_table",
                    return_value=table,
                ),
            ):
                ctx, _ = self.navi._build_zero_cut_context(
                    SimpleNamespace(),
                    SimpleNamespace(),
                    {"slice_strategy": strategy},
                )

            self.assertEqual(ctx.subc_ranges, expected_ranges)
            self.assertEqual(
                ctx.partition_candidates,
                [table[left][right] for left, right in expected_ranges],
            )

    def test_shared_prefix_signature_normalizes_greedy_and_separates_dp(self):
        implicit_greedy = self.navi.get_shared_prefix_signature({})
        explicit_dp = self.navi.get_shared_prefix_signature({"slice_strategy": "dp"})
        greedy = self.navi.get_shared_prefix_signature(
            {"slice_strategy": "greedy_table"}
        )

        self.assertEqual(implicit_greedy, greedy)
        self.assertNotEqual(explicit_dp, greedy)
        self.assertEqual(explicit_dp[-1], "dp")
        self.assertEqual(greedy[-1], "greedy_table")
        self.assertEqual(len(explicit_dp), 4)

    def test_experiment_configs_are_paired_and_output_isolated(self):
        configs = validate_paired_configs()
        for config in configs.values():
            self.assertFalse(config["compiler_config"]["navi"]["ablation_mode"])
            self.assertFalse(config["navi_checkpoint"]["enabled"])
            self.assertEqual(config["network_config"]["seed"], 42)
            self.assertEqual(
                set(config["compiler_config"]["navi"]),
                {"ablation_mode", "slice_strategy"},
            )
        self.assertEqual(
            configs["dp"]["compiler_config"]["navi"]["slice_strategy"],
            "dp",
        )
        self.assertEqual(
            configs["greedy_table"]["compiler_config"]["navi"]["slice_strategy"],
            "greedy_table",
        )
        self.assertNotEqual(
            configs["dp"]["output_folder"],
            configs["greedy_table"]["output_folder"],
        )


if __name__ == "__main__":
    unittest.main()
