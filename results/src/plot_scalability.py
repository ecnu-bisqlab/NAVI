from pathlib import Path
import ast

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.artist import Artist
import numpy as np
import pandas as pd

from .config import DEFAULT_CONFIG, PlotConfig
from .io import build_output_paths, finalize_plot_metadata


def _build_qpu_labels(df: pd.DataFrame, common_index: list[tuple[object, object]]) -> list[str]:
    qpu_label_map = {}
    key_cols = ['#Qubits', '#QPUs', 'QPUs']
    df_cfg = df[key_cols].drop_duplicates()
    for _, row in df_cfg.iterrows():
        key = (row['#Qubits'], row['#QPUs'])
        qpu_list = None
        qpus_raw = row['QPUs']
        if isinstance(qpus_raw, (list, tuple)) and len(qpus_raw) > 0:
            qpu_list = list(qpus_raw)
        elif isinstance(qpus_raw, str):
            try:
                parsed = ast.literal_eval(qpus_raw)
                if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
                    qpu_list = list(parsed)
            except (ValueError, SyntaxError):
                pass
        if qpu_list is not None:
            vals = [float(v) for v in qpu_list]
            if len(vals) > 0 and np.allclose(vals, vals[0]):
                qpu_label_map[key] = rf"{int(row['#QPUs'])}$\times${int(round(vals[0]))}"
            else:
                qpu_label_map[key] = '[' + ', '.join((str(int(round(v))) for v in vals)) + ']'
        elif row['#QPUs']:
            size_per_qpu = row['#Qubits'] / row['#QPUs']
            qpu_label_map[key] = rf"{int(row['#QPUs'])}$\times${int(round(float(size_per_qpu)))}"
        else:
            qpu_label_map[key] = str(row['#QPUs'])

    x_labels = []
    for qubits, qpus in common_index:
        label = qpu_label_map.get((qubits, qpus))
        if label is None:
            if qpus:
                size = int(round(float(qubits) / float(qpus)))
                label = rf'{int(qpus)}$\times${int(size)}'
            else:
                label = str(qpus)
        x_labels.append(label)
    return x_labels


def _fnorm_legend_handles(
    df_f: pd.DataFrame,
    df_e: pd.DataFrame,
    cfg: PlotConfig,
) -> tuple[list[Artist], list[Artist]]:
    method_order = cfg.method_order
    method_color = cfg.method_color
    method_marker = cfg.method_marker
    method_hatch = cfg.method_hatch
    line_header = mlines.Line2D([0], [0], color='black', linewidth=2.5, label='$\\mathbf{F}_{\\mathbf{norm}}$')
    line_row: list[Artist] = [line_header]
    for m in method_order:
        if m in df_f.columns:
            line_row.append(mlines.Line2D([0], [0], color=method_color[m], marker=method_marker[m], linewidth=2.4 if m == 'NAVI (Ours)' else 2.0, markersize=13 if m == 'NAVI (Ours)' else 8, markeredgecolor='black', markeredgewidth=0.8, label=m))
    bar_header = mpatches.Patch(facecolor='white', edgecolor='black', linewidth=1.5, label='EPR')
    bar_row: list[Artist] = [bar_header]
    for m in method_order:
        if m in df_e.columns:
            bar_row.append(mpatches.Patch(facecolor=method_color[m], edgecolor='black', hatch=method_hatch[m], linewidth=1.6 if m == 'NAVI (Ours)' else 1.0, label=m))
    return line_row, bar_row


def _add_fnorm_legends(
    fig,
    line_handles: list[Artist],
    bar_handles: list[Artist],
) -> tuple[Artist, Artist]:
    legend_style = dict(
        loc='upper center',
        ncol=len(line_handles),
        prop={'size': 13},
        handlelength=1.5,
        handletextpad=0.25,
        columnspacing=0.8,
        labelspacing=0.45,
        alignment='center',
        frameon=False,
    )
    line_legend = fig.legend(
        handles=line_handles,
        bbox_to_anchor=(0.28, 0.92),
        **legend_style,
    )
    bar_legend = fig.legend(
        handles=bar_handles,
        bbox_to_anchor=(0.73, 0.92),
        **legend_style,
    )
    return line_legend, bar_legend


def _plot_dual_axis_epairs_fnorm_panel(
    df_long: pd.DataFrame,
    ax_left,
    circuit: str,
    cfg: PlotConfig,
    panel_label: str | None = None,
) -> tuple[pd.DataFrame, list[Artist], list[Artist]]:
    method_order = cfg.method_order
    method_color = cfg.method_color
    method_marker = cfg.method_marker
    method_hatch = cfg.method_hatch
    df = df_long[df_long['Circuit'] == circuit].copy()
    df_f = df[df['Metrics'] == 'F_norm'].pivot_table(index=['#Qubits', '#QPUs'], columns='MethodShort', values='Value', aggfunc='first')
    df_e = df[df['Metrics'] == 'EPR Pair Consumption'].pivot_table(index=['#Qubits', '#QPUs'], columns='MethodShort', values='Value', aggfunc='first')
    common_index = df_f.index.intersection(df_e.index)
    df_f = df_f.loc[common_index]
    df_e = df_e.loc[common_index]
    common_index = sorted(common_index, key=lambda t: (t[0], t[1]))
    df_f = df_f.loc[common_index]
    df_e = df_e.loc[common_index]
    x = np.arange(len(common_index))
    ax_right = ax_left.twinx()
    ax_left.set_zorder(3)
    ax_right.set_zorder(2)
    ax_left.patch.set_visible(False)
    bar_width = 0.13
    offsets = np.linspace(-2, 2, len(method_order)) * bar_width
    for i, method in enumerate(method_order):
        if method not in df_e.columns:
            continue
        bars = ax_right.bar(x + offsets[i], df_e[method].to_numpy(dtype=float), width=bar_width, color=method_color[method], edgecolor='black', linewidth=1.0, alpha=0.9, zorder=1)
        if method == 'NAVI (Ours)':
            for b in bars:
                b.set_linewidth(1.6)
        hatch = method_hatch[method]
        if hatch:
            for b in bars:
                ax_right.add_patch(mpatches.Rectangle((b.get_x(), b.get_y()), b.get_width(), b.get_height(), facecolor='none', edgecolor='#555555', hatch=hatch, linewidth=0.0, zorder=b.get_zorder() + 0.1))
                ax_right.add_patch(mpatches.Rectangle((b.get_x(), b.get_y()), b.get_width(), b.get_height(), facecolor='none', edgecolor='black', linewidth=b.get_linewidth(), zorder=b.get_zorder() + 0.2))
    for method in method_order:
        if method not in df_f.columns:
            continue
        ax_left.plot(x, df_f[method].to_numpy(dtype=float), color=method_color[method], marker=method_marker[method], linewidth=2.4 if method == 'NAVI (Ours)' else 2.0, markersize=13 if method == 'NAVI (Ours)' else 8, markeredgecolor='black', markeredgewidth=0.8, zorder=3)
    valid_f_cols = [m for m in method_order if m in df_f.columns]
    all_f = df_f[valid_f_cols].to_numpy(dtype=float).flatten()
    all_f = all_f[~np.isnan(all_f)]
    f_min = np.min(all_f)
    f_max = np.max(all_f)
    f_span = f_max - f_min
    f_pad = max(f_span * 0.18, 0.01)
    lower = max(0.0, f_min - f_pad)
    upper = min(1.0, f_max + f_pad)
    if upper - lower < 0.08:
        center = 0.5 * (upper + lower)
        lower = max(0.0, center - 0.04)
        upper = min(1.0, center + 0.04)
    ax_left.set_ylim(lower, upper)
    ax_left.set_ylabel('$\\mathbf{F}_{\\mathbf{norm}}$')
    ax_right.set_ylabel('EPR Pair Consumption')
    x_labels = _build_qpu_labels(df, common_index)
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(x_labels)
    x_axis_label = r'#QPUs $\times$ #Qubits/QPU'
    if panel_label:
        x_axis_label = f'{x_axis_label}\n{panel_label}'
    ax_left.set_xlabel(x_axis_label, labelpad=6)
    ax_left.grid(True, axis='y', linestyle='--', linewidth=0.6, alpha=0.35, zorder=0)
    for ax in [ax_left, ax_right]:
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
    for ax in [ax_left, ax_right]:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
    line_handles, bar_handles = _fnorm_legend_handles(df_f, df_e, cfg)
    df_f_meta = df_f.reset_index().melt(id_vars=['#Qubits', '#QPUs'], value_vars=[m for m in method_order if m in df_f.columns], var_name='MethodShort', value_name='F_norm')
    df_e_meta = df_e.reset_index().melt(id_vars=['#Qubits', '#QPUs'], value_vars=[m for m in method_order if m in df_e.columns], var_name='MethodShort', value_name='EPR Pair Consumption')
    df_meta = df_f_meta.merge(df_e_meta, on=['#Qubits', '#QPUs', 'MethodShort'], how='inner')
    df_meta.insert(0, 'Circuit', circuit)
    df_meta['MethodOrder'] = df_meta['MethodShort'].map({m: i for i, m in enumerate(method_order)})
    meta = finalize_plot_metadata(df_meta, ['Circuit', '#Qubits', '#QPUs', 'MethodShort', 'MethodOrder', 'F_norm', 'EPR Pair Consumption'], sort_by=['#Qubits', '#QPUs', 'MethodOrder'])
    return meta, line_handles, bar_handles


def plot_dual_axis_epairs_fnorm(df_long, order=None, output_prefix: str | Path='random_scalability_qpus', circuit='Random', cfg: PlotConfig | None=None):
    cfg = cfg or DEFAULT_CONFIG
    fig, ax_left = plt.subplots(figsize=(8, 5.6))
    df_meta, line_handles, bar_handles = _plot_dual_axis_epairs_fnorm_panel(
        df_long=df_long,
        ax_left=ax_left,
        circuit=circuit,
        cfg=cfg,
    )
    legends = _add_fnorm_legends(fig, line_handles, bar_handles)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    _, pdf_path = build_output_paths(output_prefix)
    fig.savefig(pdf_path, bbox_inches='tight', bbox_extra_artists=legends, pad_inches=0.08)
    plt.show()
    plt.close(fig)
    return df_meta


def plot_combined_scalability_fnorm(
    df_long_scalability_qpu: pd.DataFrame,
    df_long_qubits: pd.DataFrame,
    output_prefix: str | Path='scalability',
    circuit='Random',
    cfg: PlotConfig | None=None,
) -> pd.DataFrame:
    cfg = cfg or DEFAULT_CONFIG
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.3))
    qpu_meta, line_handles, bar_handles = _plot_dual_axis_epairs_fnorm_panel(
        df_long=df_long_scalability_qpu,
        ax_left=axes[0],
        circuit=circuit,
        cfg=cfg,
        panel_label='(a)',
    )
    qubits_meta, _, _ = _plot_dual_axis_epairs_fnorm_panel(
        df_long=df_long_qubits,
        ax_left=axes[1],
        circuit=circuit,
        cfg=cfg,
        panel_label='(b)',
    )
    legends = _add_fnorm_legends(fig, line_handles, bar_handles)
    fig.tight_layout(rect=(0, 0.12, 1, 0.88), w_pad=2.0)
    _, pdf_path = build_output_paths(output_prefix)
    fig.savefig(pdf_path, bbox_inches='tight', bbox_extra_artists=legends, pad_inches=0.08)
    plt.show()
    plt.close(fig)

    qpu_meta = qpu_meta.copy()
    qubits_meta = qubits_meta.copy()
    qpu_meta.insert(0, 'Scalability', 'qpus')
    qubits_meta.insert(0, 'Scalability', 'qubits')
    return pd.concat([qpu_meta, qubits_meta], ignore_index=True)

def plot_dual_axis_epairs_error(df_long, order=None, output_prefix: str | Path='random_scalability_qpus_error', circuit='Random', cfg: PlotConfig | None=None):
    cfg = cfg or DEFAULT_CONFIG
    method_order = cfg.method_order
    method_color = cfg.method_color
    method_marker = cfg.method_marker
    method_hatch = cfg.method_hatch
    df = df_long[df_long['Circuit'] == circuit].copy()
    df_err = df[df['Metrics'] == 'Cumulative Error'].pivot_table(index=['#Qubits', '#QPUs'], columns='MethodShort', values='Value', aggfunc='first')
    df_e = df[df['Metrics'] == 'EPR Pair Consumption'].pivot_table(index=['#Qubits', '#QPUs'], columns='MethodShort', values='Value', aggfunc='first')
    common_index = df_err.index.intersection(df_e.index)
    df_err = df_err.loc[common_index]
    df_e = df_e.loc[common_index]
    common_index = sorted(common_index, key=lambda t: (t[0], t[1]))
    df_err = df_err.loc[common_index]
    df_e = df_e.loc[common_index]
    x = np.arange(len(common_index))
    fig, ax_left = plt.subplots(figsize=(8, 5.6))
    ax_right = ax_left.twinx()
    ax_left.set_zorder(3)
    ax_right.set_zorder(2)
    ax_left.patch.set_visible(False)
    bar_width = 0.13
    offsets = np.linspace(-2, 2, len(method_order)) * bar_width
    for i, method in enumerate(method_order):
        if method not in df_e.columns:
            continue
        bars = ax_right.bar(x + offsets[i], df_e[method].to_numpy(dtype=float), width=bar_width, color=method_color[method], edgecolor='black', linewidth=1.0, alpha=0.9, zorder=1)
        if method == 'NAVI (Ours)':
            for b in bars:
                b.set_linewidth(1.6)
        hatch = method_hatch[method]
        if hatch:
            for b in bars:
                ax_right.add_patch(mpatches.Rectangle((b.get_x(), b.get_y()), b.get_width(), b.get_height(), facecolor='none', edgecolor='#555555', hatch=hatch, linewidth=0.0, zorder=b.get_zorder() + 0.1))
                ax_right.add_patch(mpatches.Rectangle((b.get_x(), b.get_y()), b.get_width(), b.get_height(), facecolor='none', edgecolor='black', linewidth=b.get_linewidth(), zorder=b.get_zorder() + 0.2))
    for method in method_order:
        if method not in df_err.columns:
            continue
        ax_left.plot(x, df_err[method].to_numpy(dtype=float), color=method_color[method], marker=method_marker[method], linewidth=2.4 if method == 'NAVI (Ours)' else 2.0, markersize=13 if method == 'NAVI (Ours)' else 8, markeredgecolor='black', markeredgewidth=0.8, zorder=3)
    valid_err_cols = [m for m in method_order if m in df_err.columns]
    all_err = df_err[valid_err_cols].to_numpy(dtype=float).flatten()
    all_err = all_err[~np.isnan(all_err)]
    if all_err.size == 0:
        raise ValueError(f'No valid Cumulative Error values found for circuit={circuit!r}.')
    err_min = np.min(all_err)
    err_max = np.max(all_err)
    err_span = err_max - err_min
    err_pad = max(err_span * 0.18, max(err_max * 0.05, 1e-06))
    lower = max(0.0, err_min - err_pad)
    upper = err_max + err_pad
    if upper <= lower:
        upper = lower + max(err_max * 0.1, 1e-06)
    ax_left.set_ylim(lower, upper)
    ax_left.set_ylabel('Cumulative Error')
    ax_right.set_ylabel('EPR Pair Consumption')
    qpu_label_map = {}
    key_cols = ['#Qubits', '#QPUs', 'QPUs']
    df_cfg = df[key_cols].drop_duplicates()
    for _, row in df_cfg.iterrows():
        key = (row['#Qubits'], row['#QPUs'])
        qpu_list = None
        qpus_raw = row['QPUs']
        if isinstance(qpus_raw, (list, tuple)) and len(qpus_raw) > 0:
            qpu_list = list(qpus_raw)
        elif isinstance(qpus_raw, str):
            try:
                parsed = ast.literal_eval(qpus_raw)
                if isinstance(parsed, (list, tuple)) and len(parsed) > 0:
                    qpu_list = list(parsed)
            except (ValueError, SyntaxError):
                pass
        if qpu_list is not None:
            vals = [float(v) for v in qpu_list]
            if len(vals) > 0 and np.allclose(vals, vals[0]):
                qpu_label_map[key] = rf"{int(row['#QPUs'])}$\times${int(round(vals[0]))}"
            else:
                qpu_label_map[key] = '[' + ', '.join((str(int(round(v))) for v in vals)) + ']'
        elif row['#QPUs']:
            size_per_qpu = row['#Qubits'] / row['#QPUs']
            qpu_label_map[key] = rf"{int(row['#QPUs'])}$\times${int(round(float(size_per_qpu)))}"
        else:
            qpu_label_map[key] = str(row['#QPUs'])
    x_labels = []
    for qubits, qpus in common_index:
        label = qpu_label_map.get((qubits, qpus))
        if label is None:
            if qpus:
                size = int(round(float(qubits) / float(qpus)))
                label = rf'{int(qpus)}$\times${int(size)}'
            else:
                label = str(qpus)
        x_labels.append(label)
    ax_left.set_xticks(x)
    ax_left.set_xticklabels(x_labels)
    ax_left.set_xlabel(r'#QPUs $\times$ #Qubits/QPU')
    ax_left.grid(True, axis='y', linestyle='--', linewidth=0.6, alpha=0.35, zorder=0)
    for ax in [ax_left, ax_right]:
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
    for ax in [ax_left, ax_right]:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight('bold')
    line_header = mlines.Line2D([0], [0], color='black', linewidth=2.5, label='Error')
    total_cols = len(method_order) + 1
    blank_handle = mlines.Line2D([], [], linestyle='', marker='', alpha=0, label='')
    line_row: list[Artist] = [line_header]
    for m in method_order:
        if m in df_err.columns:
            line_row.append(mlines.Line2D([0], [0], color=method_color[m], marker=method_marker[m], linewidth=2.4 if m == 'NAVI (Ours)' else 2.0, markersize=13 if m == 'NAVI (Ours)' else 8, markeredgecolor='black', markeredgewidth=0.8, label=m))
        else:
            line_row.append(blank_handle)
    bar_header = mpatches.Patch(facecolor='white', edgecolor='black', linewidth=1.5, label='EPR')
    bar_row: list[Artist] = [bar_header]
    for m in method_order:
        if m in df_e.columns:
            bar_row.append(mpatches.Patch(facecolor=method_color[m], edgecolor='black', hatch=method_hatch[m], linewidth=1.6 if m == 'NAVI (Ours)' else 1.0, label=m))
        else:
            bar_row.append(blank_handle)
    legend_handles: list[Artist] = []
    for h_line, h_bar in zip(line_row, bar_row):
        legend_handles.extend([h_line, h_bar])
    legend = fig.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(0.0, 0.97), ncol=total_cols, prop={'size': 13}, handlelength=1.5, handletextpad=0.25, columnspacing=1, labelspacing=0.45, alignment='left', frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    _, pdf_path = build_output_paths(output_prefix)
    fig.savefig(pdf_path, bbox_inches='tight', bbox_extra_artists=(legend,))
    plt.show()
    plt.close(fig)
    df_err_meta = df_err.reset_index().melt(id_vars=['#Qubits', '#QPUs'], value_vars=[m for m in method_order if m in df_err.columns], var_name='MethodShort', value_name='Cumulative Error')
    df_e_meta = df_e.reset_index().melt(id_vars=['#Qubits', '#QPUs'], value_vars=[m for m in method_order if m in df_e.columns], var_name='MethodShort', value_name='EPR Pair Consumption')
    df_meta = df_err_meta.merge(df_e_meta, on=['#Qubits', '#QPUs', 'MethodShort'], how='inner')
    df_meta.insert(0, 'Circuit', circuit)
    df_meta['MethodOrder'] = df_meta['MethodShort'].map({m: i for i, m in enumerate(method_order)})
    return finalize_plot_metadata(df_meta, ['Circuit', '#Qubits', '#QPUs', 'MethodShort', 'MethodOrder', 'Cumulative Error', 'EPR Pair Consumption'], sort_by=['#Qubits', '#QPUs', 'MethodOrder'])

__all__ = ['plot_dual_axis_epairs_fnorm', 'plot_dual_axis_epairs_error', 'plot_combined_scalability_fnorm']
