from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.artist import Artist

from .config import PlotConfig
from .io import build_output_paths
from .style import set_pub_style


def _pick_method_columns(df: pd.DataFrame, metadata_columns: list[str]) -> list[str]:
    excluded = set(metadata_columns + ["Metrics"])
    method_cols: list[str] = []
    for col in df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            method_cols.append(col)
    return method_cols


def _metric_improvement(metric: str, anchor: pd.Series, baseline: pd.Series) -> pd.Series:
    if metric == "F_norm":
        return anchor - baseline
    denom = baseline.replace(0, np.nan)
    return (baseline - anchor) / denom * 100


def plot_network_robustness(
    input_path: str | Path,
    output_prefix: str | Path,
    config: PlotConfig,
    metadata_columns: list[str] | None = None,
    metric_order: list[str] | None = None,
    sheet_name: str | None = None,
    show_in_notebook: bool = True,
    use_pub_style: bool = True,
) -> pd.DataFrame:
    if use_pub_style:
        set_pub_style()

    input_path = Path(input_path)
    suffix = input_path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        if sheet_name is None:
            xls = pd.ExcelFile(input_path)
            sheet_name = str(xls.sheet_names[0])
        df = pd.read_excel(input_path, sheet_name=sheet_name)
    elif suffix == ".csv":
        df = pd.read_csv(input_path)
    else:
        raise ValueError(f"Unsupported file type: {input_path.suffix}")

    if "Metrics" not in df.columns:
        raise KeyError("Expected a 'Metrics' column.")
    if "net_type" not in df.columns:
        raise KeyError("Expected a 'net_type' column for topology grouping.")

    if metadata_columns is None:
        metrics_idx = list(df.columns).index("Metrics")
        metadata_columns = list(df.columns[:metrics_idx])

    method_cols = _pick_method_columns(df, metadata_columns=metadata_columns)
    if config.anchor_method not in method_cols:
        raise ValueError(f"Anchor method '{config.anchor_method}' not found in detected method columns: {method_cols}")

    baselines = [m for m in method_cols if m != config.anchor_method]
    if config.baseline_order:
        order_set = [m for m in config.baseline_order if m in baselines]
        baselines = order_set + [m for m in baselines if m not in order_set]

    metric_rules = {
        "F_norm": {"ylabel": r"$\mathbf{F_{norm}}$ Improvement", "panel": "(a)", "kind": "delta"},
        "Cumulative Error": {"ylabel": "Cumulative Error Reduction (%)", "panel": "(b)", "kind": "ratio"},
        "EPR Pair Consumption": {"ylabel": "EPR Pair Consumption Reduction (%)", "panel": "(c)", "kind": "ratio"},
    }
    metric_order = metric_order or ["F_norm", "Cumulative Error", "EPR Pair Consumption"]
    available_metric_values = set(df["Metrics"].dropna().unique())
    used_metrics = [m for m in metric_order if m in available_metric_values]
    if not used_metrics:
        raise ValueError(f"No target metrics found. Tried: {metric_order}")

    topology_order = [str(t) for t in pd.unique(df["net_type"]) if pd.notna(t)]
    topology_label_map = {
        "mesh_grid": "Mesh Grid Topology",
        "star": "Star Topology",
        "chain": "Chain Topology",
        "all_to_all": "All-to-All Topology",
    }
    palette = dict(
        zip(
            topology_order,
            sns.color_palette("Set2", n_colors=max(len(topology_order), 1))[: len(topology_order)],
        )
    )

    rows: list[dict[str, object]] = []
    for metric in used_metrics:
        dm = df[df["Metrics"] == metric].copy()
        if dm.empty:
            continue
        anchor = pd.to_numeric(dm[config.anchor_method], errors="coerce")
        rule_kind = metric_rules.get(metric, {"kind": "ratio"})["kind"]
        for base in baselines:
            baseline = pd.to_numeric(dm[base], errors="coerce")
            if rule_kind == "delta":
                imp = anchor - baseline
            else:
                denom = baseline.replace(0, np.nan)
                imp = (baseline - anchor) / denom * 100
            for i, val in imp.items():
                if pd.isna(val) or np.isinf(val):
                    continue
                rows.append(
                    {
                        "Network": str(dm.at[i, "net_type"]),
                        "Circuit": dm.at[i, "Circuit"] if "Circuit" in dm.columns else f"row_{i}",
                        "Metric": metric,
                        "Baseline": base,
                        "Improvement": float(val),
                    }
                )

    dfi = pd.DataFrame(rows)
    if dfi.empty:
        raise ValueError("No valid topology robustness comparisons were generated.")

    if config.bench_order:
        circuit_rank = {c: idx for idx, c in enumerate(config.bench_order)}
        dfi["BenchRank"] = dfi["Circuit"].map(circuit_rank).fillna(len(circuit_rank))
    else:
        dfi["BenchRank"] = 0

    fig, axes = plt.subplots(1, len(used_metrics), figsize=(18, 6.8), sharex=False)
    if len(used_metrics) == 1:
        axes = [axes]

    legend_handles: list[Artist] = []
    legend_labels: list[str] = []
    box_width = 0.72
    hue_count = max(len(topology_order), 1)
    hue_offsets = np.linspace(
        -box_width / 2 + box_width / (2 * hue_count),
        box_width / 2 - box_width / (2 * hue_count),
        hue_count,
    )
    hue_slot_width = box_width / hue_count

    for ax, metric in zip(axes, used_metrics):
        rule = metric_rules.get(metric, {"ylabel": "Improvement", "panel": ""})
        plot_df = dfi[dfi["Metric"] == metric].copy()
        if plot_df.empty:
            ax.set_visible(False)
            continue

        used_baselines = [b for b in baselines if b in plot_df["Baseline"].unique()]
        sns.boxplot(
            data=plot_df,
            x="Baseline",
            y="Improvement",
            hue="Network",
            order=used_baselines,
            hue_order=topology_order,
            palette=palette,
            width=box_width,
            linewidth=1.2,
            showfliers=False,
            ax=ax,
        )

        ax.set_xticks(np.arange(len(used_baselines)))
        ax.set_xticklabels(used_baselines)
        ax.tick_params(axis="x", pad=10)

        for base_idx, baseline in enumerate(used_baselines):
            for net_idx, network in enumerate(topology_order):
                group = plot_df[
                    (plot_df["Baseline"] == baseline) & (plot_df["Network"] == network)
                ].sort_values(["BenchRank", "Circuit"], kind="stable")
                if group.empty:
                    continue

                slot_center = base_idx + hue_offsets[net_idx]
                if len(group) == 1:
                    x_positions = np.array([slot_center])
                else:
                    spread = min(hue_slot_width * 0.62, 0.12)
                    x_positions = slot_center + np.linspace(-spread, spread, len(group))

                ax.scatter(
                    x_positions,
                    group["Improvement"].to_numpy(),
                    s=28,
                    color=palette[network],
                    alpha=0.78,
                    edgecolors="black",
                    linewidths=0.25,
                    zorder=3,
                )

        handles, labels = ax.get_legend_handles_labels()
        unique: dict[str, Artist] = {}
        for handle, label in zip(handles, labels):
            if label not in unique:
                unique[label] = handle
        if not legend_handles:
            legend_handles = list(unique.values())
            legend_labels = [str(topology_label_map.get(label, label)) for label in unique.keys()]
        if ax.legend_ is not None:
            ax.legend_.remove()

        ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_xlabel("")
        ax.set_ylabel(rule["ylabel"], fontweight="bold")
        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.25)
        plt.setp(ax.get_xticklabels(), rotation=18, ha="center", rotation_mode="anchor")
        if rule.get("panel"):
            ax.text(0.5, -0.18, rule["panel"], transform=ax.transAxes, ha="center", va="top")

    legend = None
    if legend_handles and legend_labels:
        legend = fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1),
            ncol=max(1, min(len(topology_order), 4)),
            frameon=False,
        )
    fig.tight_layout(rect=(0, 0.08, 1, 0.92), w_pad=2.5)
    _, pdf_path = build_output_paths(output_prefix)
    if legend is not None:
        fig.savefig(pdf_path, bbox_inches="tight", bbox_extra_artists=(legend,))
    else:
        fig.savefig(pdf_path, bbox_inches="tight")
    if show_in_notebook:
        plt.show()
    plt.close(fig)

    return dfi.sort_values(["Metric", "Baseline", "BenchRank", "Circuit"], kind="stable").reset_index(drop=True)


__all__ = ["plot_network_robustness"]
