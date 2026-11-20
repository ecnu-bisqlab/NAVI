#!/usr/bin/env python3
"""Plot the complete six-method NAVI ablation study."""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHOD_STYLES = OrderedDict([
    ("Naive", {"color": "#d9d9d9", "hatch": ""}),
    ("SliceOpt", {"color": "#c8b9df", "hatch": "xx"}),
    ("CommOpt+MapOpt", {"color": "#79a7d3", "hatch": "//"}),
    ("SliceOpt+MapOpt", {"color": "#79c5b6", "hatch": "\\\\"}),
    ("SliceOpt+CommOpt", {"color": "#efb366", "hatch": "oo"}),
    ("NAVI", {"color": "#d94b4b", "hatch": ".."}),
])

LEAVE_ONE_OUT_STYLES = OrderedDict([
    (
        "w/o SliceOpt",
        {"column": "CommOpt+MapOpt", "color": "#79a7d3", "hatch": "//"},
    ),
    (
        "w/o CommOpt",
        {"column": "SliceOpt+MapOpt", "color": "#79c5b6", "hatch": "\\\\"},
    ),
    (
        "w/o MapOpt",
        {"column": "SliceOpt+CommOpt", "color": "#efb366", "hatch": "oo"},
    ),
    (
        "Full NAVI",
        {"column": "NAVI", "color": "#d94b4b", "hatch": ".."},
    ),
])

METRICS = OrderedDict([
    ("F_norm", {"ylabel": r"$F_{\mathrm{norm}}\ \uparrow$", "scale": "linear"}),
    ("Cumulative Error", {"ylabel": "Cumulative Error $\\downarrow$", "scale": "log"}),
    (
        "EPR Pair Consumption",
        {"ylabel": "EPR Pair Consumption $\\downarrow$", "scale": "log"},
    ),
])

# Preserve the ordering used by the existing ablation figure.
CIRCUIT_ORDER = [
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
]


def load_values(
    path: Path,
    method_styles: OrderedDict[str, dict[str, str]] = METHOD_STYLES,
) -> dict[str, dict[str, dict[str, float]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    values: dict[str, dict[str, dict[str, float]]] = {}
    for row in rows:
        circuit = str(row["Circuit"])
        metric = str(row["Metrics"])
        if metric not in METRICS:
            continue
        values.setdefault(metric, {})[circuit] = {
            method: float(row[style.get("column", method)])
            for method, style in method_styles.items()
        }

    expected = {
        (metric, circuit)
        for metric in METRICS
        for circuit in CIRCUIT_ORDER
    }
    present = {
        (metric, circuit)
        for metric, circuits in values.items()
        for circuit in circuits
    }
    missing = sorted(expected - present)
    if missing:
        raise ValueError(f"missing metric/circuit rows in {path}: {missing}")
    return values


def plot(
    input_csv: Path,
    output_pdf: Path,
    output_png: Path,
    view: str = "complete",
) -> None:
    method_styles = (
        LEAVE_ONE_OUT_STYLES if view == "leave-one-out" else METHOD_STYLES
    )
    values = load_values(input_csv, method_styles)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelweight": "bold",
        "axes.linewidth": 0.9,
        "hatch.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    figure, axes = plt.subplots(1, 3, figsize=(24.0, 6.2))
    x = np.arange(len(CIRCUIT_ORDER), dtype=float)
    bar_width = 0.16 if view == "leave-one-out" else 0.13
    offsets = (
        np.arange(len(method_styles), dtype=float)
        - (len(method_styles) - 1) / 2
    ) * bar_width

    legend_handles = []
    for panel_index, (metric, metric_style) in enumerate(METRICS.items()):
        axis = axes[panel_index]
        for method_index, (method, style) in enumerate(method_styles.items()):
            heights = [values[metric][circuit][method] for circuit in CIRCUIT_ORDER]
            bars = axis.bar(
                x + offsets[method_index],
                heights,
                width=bar_width,
                label=method,
                color=style["color"],
                edgecolor="black",
                linewidth=0.75,
                hatch=style["hatch"],
                zorder=3,
            )
            if panel_index == 0:
                legend_handles.append(bars[0])

        axis.set_ylabel(metric_style["ylabel"], fontsize=15)
        axis.set_yscale(metric_style["scale"])
        axis.set_xticks(x)
        axis.set_xticklabels(
            CIRCUIT_ORDER,
            rotation=43,
            ha="right",
            rotation_mode="anchor",
            fontsize=10.5,
            fontweight="bold",
        )
        axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35, zorder=0)
        axis.tick_params(axis="y", labelsize=10.5)
        axis.margins(x=0.015)
        if metric == "F_norm":
            axis.set_ylim(0.0, 1.08)
        axis.text(
            0.5,
            -0.34,
            f"({chr(ord('a') + panel_index)})",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=13,
            fontweight="bold",
        )

    method_names = list(method_styles)
    # Matplotlib fills multi-row legends column-first. Reorder the six-method
    # inputs so the rendered rows still follow workflow order left to right.
    legend_order = [0, 3, 1, 4, 2, 5] if view == "complete" else list(range(4))
    figure.legend(
        [legend_handles[index] for index in legend_order],
        [method_names[index] for index in legend_order],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3 if view == "complete" else 4,
        frameon=False,
        fontsize=12.5,
        handlelength=2.2,
        handleheight=1.2,
        columnspacing=2.0,
    )
    figure.subplots_adjust(
        left=0.055,
        right=0.995,
        top=0.79 if view == "complete" else 0.84,
        bottom=0.30,
        wspace=0.24,
    )

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_pdf, bbox_inches="tight")
    figure.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--view",
        choices=("complete", "leave-one-out"),
        default="complete",
        help="plot all six methods or only the three omissions plus full NAVI",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/navi_ablation_complete.csv"),
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path("outputs/navi_ablation_complete.pdf"),
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("outputs/navi_ablation_complete.png"),
    )
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    plot(arguments.input, arguments.output_pdf, arguments.output_png, arguments.view)
