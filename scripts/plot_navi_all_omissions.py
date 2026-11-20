#!/usr/bin/env python3
"""Plot all circuits for the three NAVI leave-one-out variants."""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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
METHODS = OrderedDict([
    ("w/o SliceOpt", {"color": "#79a7d3", "hatch": "//"}),
    ("w/o CommOpt", {"color": "#79c5b6", "hatch": "\\\\"}),
    ("w/o MapOpt", {"color": "#efb366", "hatch": "oo"}),
])
METRICS = OrderedDict([
    ("F_norm", {"ylabel": r"$F_{\mathrm{norm}}\ \uparrow$", "scale": "linear"}),
    ("Cumulative Error", {"ylabel": "Cumulative Error $\\downarrow$", "scale": "log"}),
    (
        "EPR Pair Consumption",
        {"ylabel": "EPR Pair Consumption $\\downarrow$", "scale": "log"},
    ),
])
SOURCE_COLUMNS = {
    "w/o CommOpt": "SliceOpt+MapOpt",
    "w/o MapOpt": "SliceOpt+CommOpt",
}


def load_values(
    static_csv: Path,
    ablation_csv: Path,
) -> dict[str, dict[str, dict[str, float]]]:
    values = {
        metric: {circuit: {} for circuit in CIRCUIT_ORDER}
        for metric in METRICS
    }
    with static_csv.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            circuit = row["Circuit"]
            if circuit not in CIRCUIT_ORDER:
                continue
            for metric in METRICS:
                values[metric][circuit]["w/o SliceOpt"] = float(row[metric])

    with ablation_csv.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            circuit = row["Circuit"]
            metric = row["Metrics"]
            if circuit not in CIRCUIT_ORDER or metric not in METRICS:
                continue
            for method, column in SOURCE_COLUMNS.items():
                values[metric][circuit][method] = float(row[column])

    missing = [
        f"{metric}/{circuit}/{method}"
        for metric in METRICS
        for circuit in CIRCUIT_ORDER
        for method in METHODS
        if method not in values[metric][circuit]
    ]
    if missing:
        raise ValueError(f"missing ablation values: {missing}")
    return values


def write_plot_data(
    values: dict[str, dict[str, dict[str, float]]],
    output_csv: Path,
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["Circuit", "Metric", *METHODS],
        )
        writer.writeheader()
        for circuit in CIRCUIT_ORDER:
            for metric in METRICS:
                writer.writerow({
                    "Circuit": circuit,
                    "Metric": metric,
                    **values[metric][circuit],
                })


def plot(
    static_csv: Path,
    ablation_csv: Path,
    output_pdf: Path,
    output_png: Path,
    output_data_csv: Path,
) -> None:
    values = load_values(static_csv, ablation_csv)
    write_plot_data(values, output_data_csv)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelweight": "bold",
        "axes.linewidth": 0.9,
        "hatch.linewidth": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    figure, axes = plt.subplots(1, 3, figsize=(24.0, 6.1))
    x = np.arange(len(CIRCUIT_ORDER), dtype=float)
    width = 0.24
    offsets = (np.arange(len(METHODS)) - 1) * width
    legend_handles = []

    for panel_index, (metric, metric_style) in enumerate(METRICS.items()):
        axis = axes[panel_index]
        for method_index, (method, style) in enumerate(METHODS.items()):
            heights = [values[metric][circuit][method] for circuit in CIRCUIT_ORDER]
            bars = axis.bar(
                x + offsets[method_index],
                heights,
                width=width,
                label=method,
                color=style["color"],
                edgecolor="black",
                linewidth=0.75,
                hatch=style["hatch"],
                zorder=3,
            )
            if panel_index == 0:
                legend_handles.append(bars[0])

        axis.set_ylabel(metric_style["ylabel"], fontsize=14)
        axis.set_yscale(metric_style["scale"])
        if metric == "F_norm":
            axis.set_ylim(0.0, 1.08)
        axis.set_xticks(x)
        axis.set_xticklabels(
            CIRCUIT_ORDER,
            rotation=43,
            ha="right",
            rotation_mode="anchor",
            fontsize=10,
            fontweight="bold",
        )
        axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35, zorder=0)
        axis.margins(x=0.012)
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

    figure.legend(
        legend_handles,
        list(METHODS),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=3,
        frameon=False,
        fontsize=12.5,
        handlelength=2.2,
        handleheight=1.2,
        columnspacing=2.2,
    )
    figure.subplots_adjust(left=0.055, right=0.995, top=0.84, bottom=0.30, wspace=0.24)
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
        default=Path(
            "outputs/navi_static_ablation_legacy_cat_no_fullspan/variant_results.csv"
        ),
    )
    parser.add_argument(
        "--ablation-csv",
        type=Path,
        default=Path("outputs/navi_ablation_new.csv"),
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path(
            "outputs/navi_static_ablation_legacy_cat_no_fullspan/all_omissions.pdf"
        ),
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path(
            "outputs/navi_static_ablation_legacy_cat_no_fullspan/all_omissions.png"
        ),
    )
    parser.add_argument(
        "--output-data-csv",
        type=Path,
        default=Path(
            "outputs/navi_static_ablation_legacy_cat_no_fullspan/"
            "all_omissions_plot_data.csv"
        ),
    )
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    plot(
        arguments.static_csv,
        arguments.ablation_csv,
        arguments.output_pdf,
        arguments.output_png,
        arguments.output_data_csv,
    )
