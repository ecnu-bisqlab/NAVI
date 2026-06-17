from __future__ import annotations

from pathlib import Path
import warnings

import pandas as pd


DEFAULT_METADATA_COLUMNS = [
    "Circuit",
    "#Qubits",
    "#Gates",
    "#2Q Gates",
    "#Depth",
    "#QPUs",
    "QPUs",
    "net_type",
    "Metrics",
]

DEFAULT_METHOD_COLUMNS = [
    "S-OEE",
    "FGP-rOEE",
    "AutoComm",
    "WBCP",
    "NAVI (Ours)",
]

DEFAULT_METRICS = [
    "F_norm",
    "Cumulative Error",
    "EPR Pair Consumption",
]

DEFAULT_PERCENT_METRICS = {
    "Cumulative Error",
    "EPR Pair Consumption",
}

DEFAULT_NAVI_METHOD = "NAVI (Ours)"
DEFAULT_IMPROVEMENT_PREFIX = "NAVI"


def normalize_sheet_columns(
    df: pd.DataFrame,
    kept_methods: list[str],
    metadata_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Keep available metadata columns and the requested processed method columns."""
    metadata_columns = metadata_columns or DEFAULT_METADATA_COLUMNS

    out = df.copy()
    if "#2Q Gates" in out.columns and "#Gates" in out.columns:
        two_q_gates = pd.to_numeric(out["#2Q Gates"], errors="coerce")
        total_gates = pd.to_numeric(out["#Gates"], errors="coerce")
        out["2Q Fraction"] = two_q_gates / total_gates
        out.loc[total_gates == 0, "2Q Fraction"] = pd.NA

    available_metadata = [col for col in metadata_columns if col in out.columns]
    available_methods = [col for col in kept_methods if col in out.columns]

    missing_methods = [col for col in kept_methods if col not in df.columns]
    if missing_methods:
        warnings.warn(
            f"Missing method columns in sheet: {missing_methods}",
            stacklevel=2,
        )

    keep_columns = available_metadata + available_methods
    return out.loc[:, keep_columns].copy()


def filter_metrics(
    df: pd.DataFrame,
    kept_metrics: list[str],
) -> pd.DataFrame:
    """Filter rows by processed metric names."""
    if "Metrics" not in df.columns:
        raise KeyError("Expected a 'Metrics' column in the input sheet.")

    out = df[df["Metrics"].isin(kept_metrics)].copy()
    missing_metrics = sorted(set(kept_metrics) - set(out["Metrics"].dropna().unique()))
    if missing_metrics:
        warnings.warn(
            f"Missing metrics in sheet: {missing_metrics}",
            stacklevel=2,
        )
    return out


def append_navi_improvements(
    df: pd.DataFrame,
    kept_methods: list[str],
    navi_method: str,
    percent_metrics: set[str] | None = None,
    improvement_prefix: str = DEFAULT_IMPROVEMENT_PREFIX,
) -> pd.DataFrame:
    """
    Add NAVI-vs-baseline improvement columns.

    Rules:
    - F_norm uses: NAVI - baseline
    - percent_metrics use: (baseline - NAVI) / baseline * 100
    """
    percent_metrics = percent_metrics or DEFAULT_PERCENT_METRICS

    if navi_method not in df.columns:
        raise KeyError(f"NAVI method column '{navi_method}' is missing.")

    out = df.copy()

    for method in kept_methods:
        if method == navi_method or method not in out.columns:
            continue

        improvement_col = f"{improvement_prefix}_vs_{method}"

        baseline = pd.to_numeric(out[method], errors="coerce")
        navi = pd.to_numeric(out[navi_method], errors="coerce")

        out[improvement_col] = navi - baseline

        percent_mask = out["Metrics"].isin(percent_metrics)
        zero_mask = baseline == 0
        out.loc[percent_mask, improvement_col] = (
            (baseline[percent_mask] - navi[percent_mask]) / baseline[percent_mask]
        ) * 100
        out.loc[percent_mask & zero_mask, improvement_col] = pd.NA

    return out


def rename_methods_and_metrics(
    df: pd.DataFrame,
    method_rename: dict[str, str] | None = None,
    metric_rename: dict[str, str] | None = None,
    circuit_rename: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Optionally rename labels; processed inputs usually need no renaming."""
    out = df.rename(columns=method_rename or {}).copy()
    if circuit_rename and "Circuit" in out.columns:
        out["Circuit"] = out["Circuit"].replace(circuit_rename)
    if metric_rename:
        out["Metrics"] = out["Metrics"].replace(metric_rename)
    return out


def append_metric_average_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append per-metric average rows at the end of a sheet.

    The average is computed over the value columns to the right of 'Metrics',
    which includes method columns and NAVI_vs_xxx improvement columns.
    """
    if "Metrics" not in df.columns:
        return df

    column_names = [str(col) for col in df.columns]
    metrics_col_idx = column_names.index("Metrics")
    value_columns = column_names[metrics_col_idx + 1:]
    if not value_columns:
        return df

    average_rows: list[dict[str, object]] = []
    for metric_name, group in df.groupby("Metrics", sort=False):
        avg_row: dict[str, object] = {col: pd.NA for col in column_names}
        if "Circuit" in avg_row:
            avg_row["Circuit"] = "Average"
        avg_row["Metrics"] = metric_name

        for col in value_columns:
            numeric_values = pd.to_numeric(group[col], errors="coerce")
            avg_row[col] = numeric_values.mean()

        average_rows.append(avg_row)

    if not average_rows:
        return df

    # Build the appended rows from an empty slice of the original frame so the
    # column dtypes stay aligned and pandas does not need to infer concat dtypes
    # from all-NA metadata columns.
    df_average = df.iloc[0:0].copy().reindex(range(len(average_rows)))
    for row_idx, avg_row in enumerate(average_rows):
        for col, value in avg_row.items():
            df_average.at[row_idx, col] = value

    return pd.concat([df, df_average], ignore_index=True)


def sort_by_bench_order(
    df: pd.DataFrame,
    bench_order: list[str] | None = None,
) -> pd.DataFrame:
    """Sort rows by the user-provided circuit order while keeping average rows last."""
    if not bench_order or "Circuit" not in df.columns:
        return df

    out = df.copy()
    average_mask = out["Circuit"].eq("Average")
    order_map = {name: idx for idx, name in enumerate(bench_order)}

    out["_is_average"] = average_mask.astype(int)
    out["_bench_order"] = out["Circuit"].map(order_map).fillna(len(order_map))
    out = out.sort_values(
        by=["_is_average", "_bench_order"],
        kind="stable",
    ).drop(columns=["_is_average", "_bench_order"])
    out = out.reset_index(drop=True)
    return out


def analyze_sheet(
    df: pd.DataFrame,
    kept_methods: list[str] | None = None,
    kept_metrics: list[str] | None = None,
    navi_method: str = DEFAULT_NAVI_METHOD,
    percent_metrics: set[str] | None = None,
    improvement_prefix: str = DEFAULT_IMPROVEMENT_PREFIX,
    method_rename: dict[str, str] | None = None,
    metric_rename: dict[str, str] | None = None,
    metadata_columns: list[str] | None = None,
    bench_order: list[str] | None = None,
    circuit_rename: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Analyze a single sheet while preserving the wide-table layout."""
    kept_methods = kept_methods or DEFAULT_METHOD_COLUMNS
    kept_metrics = kept_metrics or DEFAULT_METRICS

    out = normalize_sheet_columns(
        df,
        kept_methods=kept_methods,
        metadata_columns=metadata_columns,
    )
    out = filter_metrics(out, kept_metrics=kept_metrics)
    out = append_navi_improvements(
        out,
        kept_methods=kept_methods,
        navi_method=navi_method,
        percent_metrics=percent_metrics,
        improvement_prefix=improvement_prefix,
    )
    out = rename_methods_and_metrics(
        out,
        method_rename=method_rename,
        metric_rename=metric_rename,
        circuit_rename=circuit_rename,
    )
    out = sort_by_bench_order(out, bench_order=bench_order)
    out = append_metric_average_rows(out)
    return out


def analyze_workbook(
    input_xlsx: str | Path,
    output_xlsx: str | Path,
    kept_methods: list[str] | None = None,
    kept_metrics: list[str] | None = None,
    navi_method: str = DEFAULT_NAVI_METHOD,
    percent_metrics: set[str] | None = None,
    improvement_prefix: str = DEFAULT_IMPROVEMENT_PREFIX,
    method_rename: dict[str, str] | None = None,
    metric_rename: dict[str, str] | None = None,
    metadata_columns: list[str] | None = None,
    bench_order: list[str] | None = None,
    circuit_rename: dict[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Analyze every sheet in a workbook and write results back sheet-by-sheet.

    Returns a dict keyed by sheet name for convenient interactive use.
    """
    input_xlsx = Path(input_xlsx)
    output_xlsx = Path(output_xlsx)
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)

    workbook = pd.ExcelFile(input_xlsx)
    analyzed_sheets: dict[str, pd.DataFrame] = {}

    with pd.ExcelWriter(output_xlsx) as writer:
        for raw_sheet_name in workbook.sheet_names:
            sheet_name = str(raw_sheet_name)
            df_sheet = pd.read_excel(input_xlsx, sheet_name=sheet_name)
            df_out = analyze_sheet(
                df_sheet,
                kept_methods=kept_methods,
                kept_metrics=kept_metrics,
                navi_method=navi_method,
                percent_metrics=percent_metrics,
                improvement_prefix=improvement_prefix,
                method_rename=method_rename,
                metric_rename=metric_rename,
                metadata_columns=metadata_columns,
                bench_order=bench_order,
                circuit_rename=circuit_rename,
            )
            df_out.to_excel(writer, sheet_name=sheet_name, index=False)
            analyzed_sheets[sheet_name] = df_out

    return analyzed_sheets


if __name__ == "__main__":
    bench_order = [
        "Random",
        "IQP",
        "QV",
        "BV",
        "MCMT",
        "Permutation",
        "Comparator",
        "QFTAdder",
        "CuccaroAdder",
        "PQC",
        "VQE",
        "SwapTest",
    ]
    jobs = [
        ("data/4x20.xlsx", "output/4x20_analysis.xlsx"),
        ("data/4x50.xlsx", "output/4x50_analysis.xlsx"),
        ("data/scalability.xlsx", "output/scalability_analysis.xlsx"),
    ]

    for input_xlsx, output_xlsx in jobs:
        if not Path(input_xlsx).exists():
            warnings.warn(f"Input workbook not found: {input_xlsx}", stacklevel=1)
            continue
        analyze_workbook(
            input_xlsx=input_xlsx,
            output_xlsx=output_xlsx,
            bench_order=bench_order,
        )
        print(f"Wrote {output_xlsx}")
