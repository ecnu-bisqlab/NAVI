#!/usr/bin/env python3
"""Plot the available static w/o SliceOpt results with the other NAVI ablations."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


CIRCUIT_ORDER = ["IQP", "QFTAdder", "BV", "SwapTest", "Comparator"]
METRICS = [
    ("F_norm", r"$F_{\mathrm{norm}}\ \uparrow$", False),
    ("Cumulative Error", "Cumulative Error $\\downarrow$", True),
    ("EPR Pair Consumption", "EPR Pair Consumption $\\downarrow$", True),
]
METHODS = [
    ("w/o SliceOpt", "#79a7d3", "//"),
    ("w/o CommOpt", "#79c5b6", "\\\\"),
    ("w/o MapOpt", "#efb366", "oo"),
    ("Full NAVI", "#d94b4b", ".."),
]

ABLATION_COLUMNS = {
    "w/o CommOpt": "SliceOpt+MapOpt",
    "w/o MapOpt": "SliceOpt+CommOpt",
    "Full NAVI": "NAVI",
}


def load_static(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        row["Circuit"]: {
            metric: float(row[metric])
            for metric, _, _ in METRICS
        }
        for row in rows
        if row["Circuit"] in CIRCUIT_ORDER
    }


def load_ablation(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = {
        method: {} for method in ABLATION_COLUMNS
    }
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            circuit = row["Circuit"]
            metric = row["Metrics"]
            if circuit in CIRCUIT_ORDER and metric in {item[0] for item in METRICS}:
                for method, column in ABLATION_COLUMNS.items():
                    result[method].setdefault(circuit, {})[metric] = float(row[column])
    return result


def format_value(metric: str, value: float) -> str:
    if metric == "F_norm":
        return f"{value:.3f}"
    if metric == "EPR Pair Consumption":
        return f"{value:.0f}"
    return f"{value:.1f}"


def plot(static_csv: Path, full_csv: Path, output_pdf: Path, output_png: Path) -> None:
    static = load_static(static_csv)
    ablation = load_ablation(full_csv)
    missing = [
        f"{method}/{circuit}"
        for method, values in [("w/o SliceOpt", static), *ablation.items()]
        for circuit in CIRCUIT_ORDER
        if circuit not in values
    ]
    if missing:
        raise ValueError(f"missing results for: {missing}")

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelweight": "bold",
        "axes.linewidth": 0.9,
        "hatch.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    figure, axes = plt.subplots(1, 3, figsize=(19.0, 5.3))
    x = np.arange(len(CIRCUIT_ORDER), dtype=float)
    width = 0.19
    values_by_method = {
        "w/o SliceOpt": static,
        **ablation,
    }

    legend_handles = []
    for panel_index, (metric, ylabel, log_scale) in enumerate(METRICS):
        axis = axes[panel_index]
        panel_bars = {}
        panel_values = {}
        for method_index, (method, color, hatch) in enumerate(METHODS):
            values = [values_by_method[method][circuit][metric] for circuit in CIRCUIT_ORDER]
            positions = x + (method_index - (len(METHODS) - 1) / 2) * width
            bars = axis.bar(
                positions,
                values,
                width=width,
                label=method,
                color=color,
                edgecolor="black",
                linewidth=0.75,
                hatch=hatch,
                zorder=3,
            )
            panel_bars[method] = bars
            panel_values[method] = values
            if panel_index == 0:
                legend_handles.append(bars[0])

        if log_scale:
            axis.set_yscale("log")
        else:
            axis.set_ylim(0.0, 1.16)
        axis.set_ylabel(ylabel, fontsize=13)
        axis.set_xticks(x)
        axis.set_xticklabels(
            CIRCUIT_ORDER,
            rotation=32,
            ha="right",
            rotation_mode="anchor",
            fontweight="bold",
        )
        axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35, zorder=0)
        axis.margins(x=0.04)

        for circuit_index, circuit in enumerate(CIRCUIT_ORDER):
            circuit_values = {
                method: panel_values[method][circuit_index]
                for method, _, _ in METHODS
            }
            best_value = (
                max(circuit_values.values())
                if metric == "F_norm"
                else min(circuit_values.values())
            )
            winners = {
                method for method, value in circuit_values.items() if value == best_value
            }
            for method, _, _ in METHODS:
                bar = panel_bars[method][circuit_index]
                value = panel_values[method][circuit_index]
                label = format_value(metric, value)
                if method in winners:
                    label += "*"
                axis.annotate(
                    label,
                    xy=(bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=7.0,
                    rotation=90,
                    fontweight="bold" if method in winners else "normal",
                )

        axis.text(
            0.5,
            -0.31,
            f"({chr(ord('a') + panel_index)})",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
        )

    figure.legend(
        legend_handles,
        [item[0] for item in METHODS],
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        frameon=False,
        fontsize=12.5,
        handlelength=2.2,
        handleheight=1.2,
        columnspacing=2.2,
    )
    figure.text(
        0.995,
        0.01,
        "* = better value for this circuit and metric",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    figure.subplots_adjust(left=0.065, right=0.99, top=0.82, bottom=0.26, wspace=0.28)

    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_pdf, bbox_inches="tight")
    figure.savefig(output_png, dpi=220, bbox_inches="tight")
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--static-csv",
        type=Path,
        default=Path("outputs/navi_static_ablation/variant_results.csv"),
    )
    parser.add_argument(
        "--full-csv",
        type=Path,
        default=Path("outputs/navi_ablation_complete.csv"),
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path("outputs/navi_static_ablation/static_leave_one_out.pdf"),
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path("outputs/navi_static_ablation/static_leave_one_out.png"),
    )
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    plot(
        arguments.static_csv,
        arguments.full_csv,
        arguments.output_pdf,
        arguments.output_png,
    )
