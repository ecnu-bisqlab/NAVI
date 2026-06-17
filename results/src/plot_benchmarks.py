from pathlib import Path
import math

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, PlotConfig
from .io import build_output_paths, finalize_plot_metadata, metric_pivot

def plot_fnorm(df_long, order=None, output_prefix: str | Path='benchmarks_fnorm', cfg: PlotConfig | None=None):
    cfg = cfg or DEFAULT_CONFIG
    method_order = cfg.method_order
    method_color = cfg.method_color
    method_hatch = cfg.method_hatch
    df = df_long[df_long['Metrics'] == 'F_norm'].copy()
    circuit_meta = df[['Circuit', '#Depth', '#2Q Gates', '#Gates']].drop_duplicates(subset=['Circuit']).set_index('Circuit')
    available_circuits = df['Circuit'].drop_duplicates().tolist()
    if order is None:
        circuit_order = available_circuits
    else:
        circuit_order = [c for c in order if c in available_circuits]
        circuit_order += [c for c in available_circuits if c not in circuit_order]
    pivot = df.pivot_table(index='Circuit', columns='MethodShort', values='Value', aggfunc='first')
    cols = [m for m in method_order if m in pivot.columns]
    pivot = pivot[cols]
    pivot = pivot.reindex([c for c in circuit_order if c in pivot.index])
    circuits = pivot.index.tolist()
    n = len(circuits)
    nrows = 2
    ncols = math.ceil(n / 2)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 4.5 * nrows), squeeze=False)
    axes = axes.flatten()

    def build_circuit_annotation(circuit):
        meta = circuit_meta.loc[circuit]
        depth = pd.to_numeric(meta['#Depth'], errors='coerce')
        depth_text = 'NA' if pd.isna(depth) else f'{int(depth)}'
        return f'$T={depth_text}$'
    for idx, (ax, circuit) in enumerate(zip(axes, circuits)):
        show_ylabel = idx % ncols == 0
        vals = pivot.loc[circuit, cols].astype(float)
        annotation_text = build_circuit_annotation(circuit)
        x = np.arange(len(cols)) * 0.6
        x = x - np.mean(x)

        def draw_bars(target_ax):
            bars = target_ax.bar(x, vals.values, width=0.35, color=[method_color[m] for m in cols], edgecolor='black', linewidth=1.0)
            for bar, method in zip(bars, cols):
                if method == 'NAVI (Ours)':
                    bar.set_linewidth(1.8)
                hatch = method_hatch[method]
                if hatch:
                    target_ax.add_patch(mpatches.Rectangle((bar.get_x(), bar.get_y()), bar.get_width(), bar.get_height(), facecolor='none', edgecolor='#555555', hatch=hatch, linewidth=0.0, zorder=bar.get_zorder() + 0.1))
                    target_ax.add_patch(mpatches.Rectangle((bar.get_x(), bar.get_y()), bar.get_width(), bar.get_height(), facecolor='none', edgecolor='black', linewidth=bar.get_linewidth(), zorder=bar.get_zorder() + 0.2))
            return bars
        draw_bars(ax)
        ax.set_xticks([])
        ax.set_title(circuit)
        ax.set_ylabel('$\\mathbf{F}_{\\mathbf{norm}}$' if show_ylabel else '')
        ax.text(0.5, 0.97, annotation_text, transform=ax.transAxes, ha='center', va='top', fontsize=16, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.85, edgecolor='none'))
        y_min = vals.min()
        y_max = vals.max()
        span = y_max - y_min
        if span < 1e-06:
            pad = max(0.01, y_max * 0.02)
        else:
            pad = max(span * 0.2, 0.01)
        lower = max(0, y_min - pad)
        upper = min(1.05, y_max + pad)
        if upper - lower < 0.03:
            center = 0.5 * (upper + lower)
            lower = max(0, center - 0.015)
            upper = min(1.05, center + 0.015)
        ax.set_ylim(lower, upper)
        ax.grid(True, axis='y', linestyle='--', linewidth=0.5, alpha=0.3)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
    for i in range(len(circuits), len(axes)):
        axes[i].axis('off')
    handles = [mpatches.Patch(facecolor=method_color[m], edgecolor='black', hatch=method_hatch[m], linewidth=1.8 if m == 'NAVI (Ours)' else 1.0, label=m) for m in cols]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=len(cols), frameon=True)
    fig.subplots_adjust(left=0.05, right=0.995, top=0.85, bottom=0.12, wspace=0.4)
    _, pdf_path = build_output_paths(output_prefix)
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    df_meta = pivot.reset_index().copy()
    df_meta.insert(1, 'Metric', 'F_norm')
    df_meta.insert(1, 'CircuitOrder', np.arange(len(df_meta)))
    return finalize_plot_metadata(df_meta, ['Circuit', 'CircuitOrder', 'Metric'] + cols, sort_by=['CircuitOrder'])

def plot_pareto_scatter(df_long_filtered, order=None, output_prefix: str | Path='per_circuit_epr_pair_consumption_pareto', x_metric='EPR Pair Consumption', y_metric='Cumulative Error', x_label='EPR Pair Consumption', y_label='Cumulative Error', ncols=4, add_arrows=False, show_dominance=True, shade_dominated_region=True, highlight_navi=True, cfg: PlotConfig | None=None):
    cfg = cfg or DEFAULT_CONFIG
    method_order = cfg.method_order
    method_color = cfg.method_color
    method_marker = cfg.method_marker
    """
    Per-circuit EPR-pair vs cumulative-error scatter plot.
    """
    df_plot = metric_pivot(df_long_filtered, [x_metric, y_metric])
    if df_plot.empty:
        print('No valid data found for plotting.')
        return None
    available_circuits = df_plot['Circuit'].drop_duplicates().tolist()
    if order is None:
        circuit_order = available_circuits
    else:
        circuit_order = [c for c in order if c in available_circuits]
        circuit_order += [c for c in available_circuits if c not in circuit_order]
    circuits = [c for c in circuit_order if c in set(df_plot['Circuit'].unique())]
    if len(circuits) == 0:
        print('No valid circuits found for plotting.')
        return None
    nrows = math.ceil(len(circuits) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.8 * ncols, 4.4 * nrows), squeeze=False)
    axes = axes.flatten()
    dominance_summary = []
    for idx, (ax, circuit) in enumerate(zip(axes, circuits)):
        sub = df_plot[df_plot['Circuit'] == circuit].copy()
        if sub.empty:
            ax.axis('off')
            continue
        row_idx = idx // ncols
        col_idx = idx % ncols
        show_xlabel = row_idx == nrows - 1
        show_ylabel = col_idx == 0
        ordered_parts = [sub[sub['MethodShort'] == method] for method in method_order]
        remaining = sub[~sub['MethodShort'].isin(method_order)]
        sub = pd.concat(ordered_parts + [remaining], ignore_index=True)
        plot_axes = [ax]
        navi = sub[sub['MethodShort'] == 'NAVI (Ours)']
        navi_x = navi_y = x_max = y_max = None
        if len(navi) == 1:
            navi_x = float(navi[x_metric].iloc[0])
            navi_y = float(navi[y_metric].iloc[0])
            x_max = float(sub[x_metric].max())
            y_max = float(sub[y_metric].max())
            if highlight_navi:
                for target_ax in plot_axes:
                    target_ax.axvline(navi_x, linestyle=':', color='#d62728', linewidth=1.5, alpha=0.85)
                    target_ax.axhline(navi_y, linestyle=':', color='#d62728', linewidth=1.5, alpha=0.85)
            dominated = 0
            baseline_total = 0
            for _, row in sub.iterrows():
                method = row['MethodShort']
                if method == 'NAVI (Ours)':
                    continue
                baseline_total += 1
                bx = float(row[x_metric])
                by = float(row[y_metric])
                is_dominated = navi_x <= bx and navi_y <= by and (navi_x < bx or navi_y < by)
                dominated += int(is_dominated)
                if add_arrows:
                    plot_axes[0].annotate('', xy=(navi_x, navi_y), xytext=(bx, by), arrowprops=dict(arrowstyle='->', color='0.45', lw=1.0, alpha=0.35, shrinkA=6, shrinkB=8), zorder=1)
            if show_dominance:
                ax.text(0.96, 0.06, f'Pareto dom. {dominated}/{baseline_total}', transform=ax.transAxes, ha='right', va='bottom', fontsize=mpl.rcParams['xtick.labelsize'], fontweight='bold')
            dominance_summary.append({'Circuit': circuit, f'NAVI_{x_metric}': navi_x, f'NAVI_{y_metric}': navi_y, 'Pareto_dominated_baselines': dominated, 'Total_baselines': baseline_total, 'Dominance_ratio': dominated / baseline_total if baseline_total > 0 else np.nan})
        for _, row in sub.iterrows():
            method = row['MethodShort']
            color = method_color.get(method, 'gray')
            marker = method_marker.get(method, 'o')
            if method == 'NAVI (Ours)':
                size = 240
                linewidth = 1.4
                zorder = 4
            else:
                size = 105
                linewidth = 0.9
                zorder = 3
            for target_ax in plot_axes:
                target_ax.scatter(row[x_metric], row[y_metric], s=size, color=color, marker=marker, edgecolor='black', linewidth=linewidth, zorder=zorder)
        ax.set_title(circuit)
        if show_xlabel:
            ax.set_xlabel(x_label)
        else:
            ax.set_xlabel('')
        if show_ylabel:
            ax.set_ylabel(y_label)
        else:
            ax.set_ylabel('')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.28)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
        if shade_dominated_region and navi_x is not None and navi_y is not None and x_max is not None and y_max is not None and x_max > navi_x and y_max > navi_y:
            x_right = ax.get_xlim()[1]
            y_top = ax.get_ylim()[1]
            if x_right > navi_x and y_top > navi_y:
                ax.fill_between([navi_x, x_right], navi_y, y_top, alpha=0.05, color='#d62728', linewidth=0, zorder=0)
    for j in range(len(circuits), len(axes)):
        axes[j].axis('off')
    handles = []
    for method in method_order:
        color = method_color.get(method, 'gray')
        marker = method_marker.get(method, 'o')
        handles.append(mlines.Line2D([], [], color=color, marker=marker, linestyle='None', markersize=10 if method != 'NAVI (Ours)' else 14, markeredgecolor='black', label=method))
    fig.legend(handles=handles, loc='upper center', ncol=len(method_order), frameon=True, bbox_to_anchor=(0.5, 0.995))
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    pdf_path = Path(output_prefix).with_suffix('.pdf')
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches='tight')
    plt.show()
    dominance_df = pd.DataFrame(dominance_summary)
    circuit_rank = {c: idx for idx, c in enumerate(circuits)}
    if not dominance_df.empty:
        dominance_df['CircuitOrder'] = dominance_df['Circuit'].map(circuit_rank)
        dominance_df = finalize_plot_metadata(dominance_df, ['Circuit', 'CircuitOrder', f'NAVI_{x_metric}', f'NAVI_{y_metric}', 'Pareto_dominated_baselines', 'Total_baselines', 'Dominance_ratio'], sort_by=['CircuitOrder'])
    return dominance_df

__all__ = ['plot_fnorm', 'plot_pareto_scatter']
