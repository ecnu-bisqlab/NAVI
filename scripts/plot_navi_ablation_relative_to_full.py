#!/usr/bin/env python3
"""Plot three NAVI ablations as direction-normalized ratios to Full NAVI."""

from __future__ import annotations

import argparse
import csv
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


METHODS = [
    ("w/o SliceOpt", "#79a7d3", "//"),
    ("w/o CommOpt", "#79c5b6", "\\\\"),
    ("w/o MapOpt", "#efb366", "oo"),
]
METRICS = [
    ("F_norm", r"$F_{\mathrm{norm}}$"),
    ("Cumulative Error", "Cumulative Error"),
    ("EPR Pair Consumption", "EPR Pair Consumption"),
]
ABLATION_COLUMNS = {
    "w/o CommOpt": "SliceOpt+MapOpt",
    "w/o MapOpt": "SliceOpt+CommOpt",
    "Full NAVI": "Full NAVI",
}


def load_static(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {
            row["Circuit"]: {
                metric: float(row[metric])
                for metric, _ in METRICS
            }
            for row in csv.DictReader(handle)
            if row["Circuit"] in CIRCUIT_ORDER
        }


def load_ablation(path: Path) -> dict[str, dict[str, dict[str, float]]]:
    result = {method: {} for method in ABLATION_COLUMNS}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            circuit = row["Circuit"]
            metric = row["Metrics"]
            if circuit not in CIRCUIT_ORDER or metric not in dict(METRICS):
                continue
            for method, column in ABLATION_COLUMNS.items():
                result[method].setdefault(circuit, {})[metric] = float(row[column])
    return result


def oriented_ratio(metric: str, ablation: float, full: float) -> float:
    """Return a ratio where values above one always mean better than Full."""
    return ablation / full if metric == "F_norm" else full / ablation


def ratio_label(ratio: float) -> str:
    if ratio >= 0.995:
        return f"{ratio:.3f}x"
    if ratio >= 0.1:
        return f"{ratio:.2f}x"
    return f"{ratio:.3f}x"


def plot(static_csv: Path, full_csv: Path, output_pdf: Path, output_png: Path) -> None:
    static = load_static(static_csv)
    existing = load_ablation(full_csv)
    values_by_method = {
        "w/o SliceOpt": static,
        "w/o CommOpt": existing["w/o CommOpt"],
        "w/o MapOpt": existing["w/o MapOpt"],
    }
    full = existing["Full NAVI"]

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
    legend_handles = []

    for panel_index, (metric, title) in enumerate(METRICS):
        axis = axes[panel_index]
        all_scores: list[float] = []
        plotted: list[tuple[object, list[float], list[float]]] = []
        for method_index, (method, color, hatch) in enumerate(METHODS):
            ratios = [
                oriented_ratio(
                    metric,
                    values_by_method[method][circuit][metric],
                    full[circuit][metric],
                )
                for circuit in CIRCUIT_ORDER
            ]
            scores = np.log2(ratios).tolist()
            positions = x + (method_index - 1) * width
            bars = axis.bar(
                positions,
                scores,
                width=width,
                label=method,
                color=color,
                edgecolor="black",
                linewidth=0.75,
                hatch=hatch,
                zorder=3,
            )
            plotted.append((bars, ratios, scores))
            all_scores.extend(scores)
            if panel_index == 0:
                legend_handles.append(bars[0])

        lower_data = min(min(all_scores) - 0.35, -0.35)
        upper_data = max(max(all_scores) + 0.35, 0.35)
        tick_min = int(np.floor(lower_data))
        tick_max = int(np.ceil(upper_data))
        lower = float(tick_min)
        upper = float(tick_max)
        axis.set_ylim(lower, upper)
        axis.axhspan(0, upper, color="#e7f4e4", alpha=0.7, zorder=0)
        axis.axhspan(lower, 0, color="#f8e5e3", alpha=0.7, zorder=0)
        axis.axhline(0, color="black", linewidth=1.1, zorder=2)

        for bars, ratios, scores in plotted:
            for bar, ratio, score in zip(bars, ratios, scores):
                offset = 3 if score >= 0 else -3
                axis.annotate(
                    ratio_label(ratio),
                    xy=(bar.get_x() + bar.get_width() / 2, score),
                    xytext=(0, offset),
                    textcoords="offset points",
                    ha="center",
                    va="bottom" if score >= 0 else "top",
                    fontsize=7.2,
                    rotation=90,
                )

        ticks = np.arange(tick_min, tick_max + 1, dtype=float)
        axis.set_yticks(ticks)
        axis.set_yticklabels([f"{2 ** tick:g}x" for tick in ticks])
        axis.set_title(title, fontsize=13, fontweight="bold", pad=8)
        axis.set_ylabel("Performance relative to Full NAVI", fontsize=11.5)
        axis.set_xticks(x)
        axis.set_xticklabels(
            CIRCUIT_ORDER,
            rotation=43,
            ha="right",
            rotation_mode="anchor",
            fontsize=10,
            fontweight="bold",
        )
        axis.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35, zorder=1)
        axis.margins(x=0.035)
        axis.text(
            0.99,
            0.97,
            "better than Full" if panel_index == 2 else "better",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            color="#3f6f3b",
        )
        axis.text(
            0.99,
            0.03,
            "worse than Full" if panel_index == 2 else "worse",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            color="#9a4941",
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
        ncol=3,
        frameon=False,
        fontsize=12.5,
        handlelength=2.2,
        handleheight=1.2,
        columnspacing=2.2,
    )
    figure.text(
        0.995,
        0.01,
        "Full NAVI = 1x; ratios are direction-normalized so higher is always better",
        ha="right",
        va="bottom",
        fontsize=9,
        color="#444444",
    )
    figure.subplots_adjust(left=0.07, right=0.99, top=0.81, bottom=0.26, wspace=0.28)

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
        "--full-csv",
        type=Path,
        default=Path("outputs/navi_ablation_new.csv"),
    )
    parser.add_argument(
        "--output-pdf",
        type=Path,
        default=Path(
            "outputs/navi_static_ablation_legacy_cat_no_fullspan/relative_to_full.pdf"
        ),
    )
    parser.add_argument(
        "--output-png",
        type=Path,
        default=Path(
            "outputs/navi_static_ablation_legacy_cat_no_fullspan/relative_to_full.png"
        ),
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
