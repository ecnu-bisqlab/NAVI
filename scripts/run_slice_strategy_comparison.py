#!/usr/bin/env python3
"""Run paired full-NAVI DP and greedy-table SliceOpt experiments."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
STRATEGY_CONFIGS = {
    "dp": REPO_ROOT / "config.slice_strategy_dp.json",
    "greedy_table": REPO_ROOT / "config.slice_strategy_greedy_table.json",
}
ALLOWED_NAVI_CONFIG_KEYS = {"ablation_mode", "slice_strategy"}
BENCHMARKS = (
    "BV",
    "SwapTest",
    "Permutation",
    "MCMT",
    "Comparator",
    "QFTAdder",
    "CuccaroAdder",
    "PQC",
    "IQP",
    "Random",
    "QV",
    "VQE",
)
METRICS = (
    "F_norm",
    "Cumulative Error",
    "EPR Pair Consumption",
    "Execution Time",
)
SUMMARY_FIELDS = (
    "Slice Strategy",
    "Circuit",
    "Segment Count",
    "Slice Planning Time",
    *METRICS,
    "Results CSV",
    "Log File",
)
COMPARISON_FIELDS = (
    "Circuit",
    "DP Segment Count",
    "Greedy Segment Count",
    "Segment Count Delta",
    "DP Slice Planning Time",
    "Greedy Slice Planning Time",
    "Slice Planning Time Delta",
    "Slice Planning Speedup",
    "DP Execution Time",
    "Greedy Execution Time",
    "Execution Time Delta",
    "Execution Time Delta Percent",
    "Execution Time Speedup",
    "DP F_norm",
    "Greedy F_norm",
    "F_norm Delta",
    "DP Cumulative Error",
    "Greedy Cumulative Error",
    "Cumulative Error Delta",
    "Cumulative Error Delta Percent",
    "DP EPR Pair Consumption",
    "Greedy EPR Pair Consumption",
    "EPR Pair Consumption Delta",
    "EPR Pair Consumption Delta Percent",
)
SLICE_LOG_PATTERN = re.compile(
    r"\[slice_planner\] strategy=(?P<strategy>[^,]+), "
    r"layers=(?P<layers>\d+), segments=(?P<segments>\d+), "
    r"time_sec=(?P<time>[0-9.]+)"
)


def _csv_values(raw: str, available: tuple[str, ...], label: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(available)
    values = [value.strip() for value in raw.split(",") if value.strip()]
    unknown = [value for value in values if value not in available]
    if unknown:
        raise ValueError(f"unknown {label}: {unknown}; available={list(available)}")
    if not values:
        raise ValueError(f"at least one {label} must be selected")
    return values


def _load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"configuration must be a JSON object: {path}")
    return config


def _comparison_payload(config: dict[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(config)
    payload.pop("experiment_id", None)
    payload.pop("output_folder", None)
    payload["compiler_config"]["navi"].pop("slice_strategy", None)
    return payload


def validate_paired_configs() -> dict[str, dict[str, Any]]:
    configs = {
        strategy: _load_config(path)
        for strategy, path in STRATEGY_CONFIGS.items()
    }
    for strategy, config in configs.items():
        configured = config.get("compiler_config", {}).get("navi", {}).get(
            "slice_strategy"
        )
        if configured != strategy:
            raise ValueError(
                f"{STRATEGY_CONFIGS[strategy]} declares slice_strategy={configured!r}; "
                f"expected {strategy!r}"
            )
        navi_config = config.get("compiler_config", {}).get("navi", {})
        if bool(navi_config.get("ablation_mode", False)):
            raise ValueError(
                f"{STRATEGY_CONFIGS[strategy]} must run full NAVI with "
                "ablation_mode=false"
            )
        unexpected_navi_keys = set(navi_config) - ALLOWED_NAVI_CONFIG_KEYS
        if unexpected_navi_keys:
            raise ValueError(
                f"{STRATEGY_CONFIGS[strategy]} overrides full-NAVI defaults: "
                f"{sorted(unexpected_navi_keys)}"
            )
        if bool(config.get("navi_checkpoint", {}).get("enabled", False)):
            raise ValueError(
                f"{STRATEGY_CONFIGS[strategy]} must disable ablation checkpoints"
            )
        expected_output = f"./outputs/slice_strategy_comparison/{strategy}/"
        if config.get("output_folder") != expected_output:
            raise ValueError(
                f"{STRATEGY_CONFIGS[strategy]} output_folder must be {expected_output!r}"
            )
    if _comparison_payload(configs["dp"]) != _comparison_payload(
        configs["greedy_table"]
    ):
        raise ValueError(
            "paired configs may differ only in experiment_id, output_folder, "
            "and slice_strategy"
        )
    return configs


def _output_dir(config: dict[str, Any]) -> Path:
    configured = Path(str(config["output_folder"]))
    return configured if configured.is_absolute() else REPO_ROOT / configured


def _read_result_metrics(path: Path, benchmark: str) -> tuple[dict[str, str], bool]:
    if not path.exists():
        return {}, False
    metrics: dict[str, str] = {}
    has_any_rows = False
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("Circuit") != benchmark:
                continue
            has_any_rows = True
            metric = str(row.get("Metrics", ""))
            value = row.get("NAVI")
            if metric in METRICS and value not in (None, ""):
                metrics[metric] = str(value)
    return metrics, has_any_rows


def _find_slice_audit(
    output_dir: Path,
    benchmark: str,
    strategy: str,
) -> tuple[int, float, Path] | None:
    logs = sorted(
        output_dir.glob(f"{benchmark}*/*.txt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in logs:
        matches = list(SLICE_LOG_PATTERN.finditer(path.read_text(encoding="utf-8")))
        for match in reversed(matches):
            if match.group("strategy") == strategy:
                return int(match.group("segments")), float(match.group("time")), path
    return None


def _load_summary(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (str(row["Slice Strategy"]), str(row["Circuit"])): dict(row)
            for row in csv.DictReader(handle)
        }


def _write_summary(
    path: Path,
    rows: dict[tuple[str, str], dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        for strategy in STRATEGY_CONFIGS:
            for benchmark in BENCHMARKS:
                row = rows.get((strategy, benchmark))
                if row is not None:
                    writer.writerow(row)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _float(row: dict[str, str], field: str) -> float:
    return float(row[field])


def _delta_percent(baseline: float, candidate: float) -> float:
    if baseline == 0.0:
        return 0.0 if candidate == 0.0 else float("inf")
    return 100.0 * (candidate - baseline) / baseline


def _speedup(baseline: float, candidate: float) -> float:
    if candidate == 0.0:
        return float("inf")
    return baseline / candidate


def _write_comparison(
    path: Path,
    rows: dict[tuple[str, str], dict[str, str]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_FIELDS)
        writer.writeheader()
        for benchmark in BENCHMARKS:
            dp = rows.get(("dp", benchmark))
            greedy = rows.get(("greedy_table", benchmark))
            if dp is None or greedy is None:
                continue

            dp_segments = int(dp["Segment Count"])
            greedy_segments = int(greedy["Segment Count"])
            dp_slice_time = _float(dp, "Slice Planning Time")
            greedy_slice_time = _float(greedy, "Slice Planning Time")
            dp_execution_time = _float(dp, "Execution Time")
            greedy_execution_time = _float(greedy, "Execution Time")
            dp_f_norm = _float(dp, "F_norm")
            greedy_f_norm = _float(greedy, "F_norm")
            dp_error = _float(dp, "Cumulative Error")
            greedy_error = _float(greedy, "Cumulative Error")
            dp_epairs = _float(dp, "EPR Pair Consumption")
            greedy_epairs = _float(greedy, "EPR Pair Consumption")

            writer.writerow({
                "Circuit": benchmark,
                "DP Segment Count": dp_segments,
                "Greedy Segment Count": greedy_segments,
                "Segment Count Delta": greedy_segments - dp_segments,
                "DP Slice Planning Time": dp_slice_time,
                "Greedy Slice Planning Time": greedy_slice_time,
                "Slice Planning Time Delta": greedy_slice_time - dp_slice_time,
                "Slice Planning Speedup": _speedup(dp_slice_time, greedy_slice_time),
                "DP Execution Time": dp_execution_time,
                "Greedy Execution Time": greedy_execution_time,
                "Execution Time Delta": greedy_execution_time - dp_execution_time,
                "Execution Time Delta Percent": _delta_percent(
                    dp_execution_time,
                    greedy_execution_time,
                ),
                "Execution Time Speedup": _speedup(
                    dp_execution_time,
                    greedy_execution_time,
                ),
                "DP F_norm": dp_f_norm,
                "Greedy F_norm": greedy_f_norm,
                "F_norm Delta": greedy_f_norm - dp_f_norm,
                "DP Cumulative Error": dp_error,
                "Greedy Cumulative Error": greedy_error,
                "Cumulative Error Delta": greedy_error - dp_error,
                "Cumulative Error Delta Percent": _delta_percent(dp_error, greedy_error),
                "DP EPR Pair Consumption": dp_epairs,
                "Greedy EPR Pair Consumption": greedy_epairs,
                "EPR Pair Consumption Delta": greedy_epairs - dp_epairs,
                "EPR Pair Consumption Delta Percent": _delta_percent(
                    dp_epairs,
                    greedy_epairs,
                ),
            })
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _command(
    python: str,
    config_path: Path,
    benchmark: str,
) -> list[str]:
    return [
        python,
        str(REPO_ROOT / "main.py"),
        "-gconf",
        str(config_path),
        "-cname",
        benchmark,
        "-nq",
        "80",
        "-core",
        "4",
        "-cap",
        "20",
        "-net",
        "mesh_grid",
        "-ocsv",
        "results.csv",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategies", default="all")
    parser.add_argument("--benchmarks", default="all")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--summary",
        type=Path,
        default=REPO_ROOT / "outputs/slice_strategy_comparison/summary.csv",
    )
    parser.add_argument(
        "--comparison",
        type=Path,
        default=REPO_ROOT / "outputs/slice_strategy_comparison/comparison.csv",
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configs = validate_paired_configs()
    strategies = _csv_values(
        args.strategies,
        tuple(STRATEGY_CONFIGS),
        "strategies",
    )
    benchmarks = _csv_values(args.benchmarks, BENCHMARKS, "benchmarks")

    if args.validate_only:
        for strategy in strategies:
            for benchmark in benchmarks:
                print(" ".join(_command(args.python, STRATEGY_CONFIGS[strategy], benchmark)))
        print("[SLICE_STRATEGY_VALIDATE] paired configs and commands are valid")
        return 0

    summary_path = args.summary
    if not summary_path.is_absolute():
        summary_path = REPO_ROOT / summary_path
    comparison_path = args.comparison
    if not comparison_path.is_absolute():
        comparison_path = REPO_ROOT / comparison_path
    summary = _load_summary(summary_path)

    for strategy in strategies:
        output_dir = _output_dir(configs[strategy])
        results_path = output_dir / "results.csv"
        for benchmark in benchmarks:
            metrics, has_any_rows = _read_result_metrics(results_path, benchmark)
            audit = _find_slice_audit(output_dir, benchmark, strategy)
            complete = set(metrics) == set(METRICS) and audit is not None
            if complete:
                print(f"[SLICE_STRATEGY_SKIP] {strategy}/{benchmark}: verified")
            else:
                if has_any_rows:
                    raise RuntimeError(
                        f"partial result exists for {strategy}/{benchmark} in {results_path}; "
                        "refusing to append duplicate rows"
                    )
                print(f"[SLICE_STRATEGY_RUN] {strategy}/{benchmark}")
                subprocess.run(
                    _command(args.python, STRATEGY_CONFIGS[strategy], benchmark),
                    cwd=REPO_ROOT,
                    check=True,
                )
                metrics, _ = _read_result_metrics(results_path, benchmark)
                audit = _find_slice_audit(output_dir, benchmark, strategy)
                if set(metrics) != set(METRICS) or audit is None:
                    raise RuntimeError(
                        f"run did not produce complete metrics and slice audit for "
                        f"{strategy}/{benchmark}"
                    )

            assert audit is not None
            segment_count, planning_time, log_path = audit
            summary[(strategy, benchmark)] = {
                "Slice Strategy": strategy,
                "Circuit": benchmark,
                "Segment Count": str(segment_count),
                "Slice Planning Time": repr(planning_time),
                **metrics,
                "Results CSV": str(results_path),
                "Log File": str(log_path),
            }
            _write_summary(summary_path, summary)
            _write_comparison(comparison_path, summary)

    print(
        f"[SLICE_STRATEGY_DONE] summary={summary_path}, "
        f"comparison={comparison_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
