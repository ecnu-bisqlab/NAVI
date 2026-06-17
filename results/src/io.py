from pathlib import Path
from typing import Literal

import pandas as pd

from .config import DEFAULT_CONFIG

ExcelWriteMode = Literal['a', 'w']
IfSheetExists = Literal['error', 'new', 'replace', 'overlay']


def read_excel_data(input_file: str | Path, excel_sheet_name: str, aggregate_runs: bool = True) -> pd.DataFrame:
    """
    Read one Excel sheet and normalize method columns.

    If aggregate_runs=True and sheets named ``<sheet>_1``, ``<sheet>_2``, ...
    exist, those sheets are averaged by metadata columns.
    """
    input_file = Path(input_file)
    required_meta = DEFAULT_CONFIG.required_meta
    method_cols = DEFAULT_CONFIG.method_cols
    circuit_rename = DEFAULT_CONFIG.circuit_rename

    def _read_one_sheet(sheet_name: str) -> pd.DataFrame:
        df = pd.read_excel(input_file, sheet_name=sheet_name, usecols=required_meta + method_cols)
        if 'Circuit' in df.columns:
            df['Circuit'] = df['Circuit'].replace(circuit_rename)
        for col in method_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=method_cols, how='all').reset_index(drop=True)
        return df

    xls = pd.ExcelFile(input_file)
    all_sheets: list[str] = [str(s) for s in xls.sheet_names]
    prefix = f'{excel_sheet_name}_'
    run_sheets: list[tuple[int, str]] = []
    for sheet_name in all_sheets:
        if sheet_name.startswith(prefix):
            suffix = sheet_name[len(prefix):]
            if suffix.isdigit():
                run_sheets.append((int(suffix), sheet_name))
    if aggregate_runs and run_sheets:
        sorted_run_sheets = [sheet_name for _, sheet_name in sorted(run_sheets, key=lambda t: t[0])]
        dfs = [_read_one_sheet(sheet_name) for sheet_name in sorted_run_sheets]
        merged = pd.concat(dfs, ignore_index=True)
        return merged.groupby(required_meta, as_index=False, sort=False)[method_cols].mean()
    if excel_sheet_name not in all_sheets:
        raise ValueError(f'Sheet not found: {excel_sheet_name}. Available sheets: {all_sheets}')
    return _read_one_sheet(excel_sheet_name)


def read_csv_data(input_file: str | Path, usecols: list[str] | None = None) -> pd.DataFrame:
    """
    Read a CSV file with the same normalization behavior as read_excel_data.

    - usecols=None: use DEFAULT_CONFIG.required_meta + DEFAULT_CONFIG.method_cols
    - Circuit names are normalized by DEFAULT_CONFIG.circuit_rename
    - method columns are converted to numeric
    - rows with all method columns as NaN are dropped
    """
    input_file = Path(input_file)
    required_meta = DEFAULT_CONFIG.required_meta
    method_cols = DEFAULT_CONFIG.method_cols
    circuit_rename = DEFAULT_CONFIG.circuit_rename
    if not input_file.exists():
        raise FileNotFoundError(f'Input CSV not found: {input_file}')
    selected_cols = usecols or required_meta + method_cols
    df = pd.read_csv(input_file, usecols=selected_cols)
    if 'Circuit' in df.columns:
        df['Circuit'] = df['Circuit'].replace(circuit_rename)
    for col in method_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    keep_for_drop = [c for c in method_cols if c in df.columns]
    if keep_for_drop:
        df = df.dropna(subset=keep_for_drop, how='all').reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    return df


def to_long(df_wide: pd.DataFrame) -> pd.DataFrame:
    required_meta = DEFAULT_CONFIG.required_meta
    method_cols = DEFAULT_CONFIG.method_cols
    method_rename = DEFAULT_CONFIG.method_rename
    df_long = df_wide.melt(id_vars=required_meta, value_vars=method_cols, var_name='Method', value_name='Value')
    df_long['MethodShort'] = df_long['Method'].map(method_rename).fillna(df_long['Method'])
    return df_long


def metric_pivot(df_long: pd.DataFrame, metric_names: list[str]) -> pd.DataFrame:
    sub = df_long[df_long['Metrics'].isin(metric_names)].copy()
    out = sub.pivot_table(index=['Circuit', 'MethodShort'], columns='Metrics', values='Value', aggfunc='first', sort=False).reset_index()
    return out


def finalize_plot_metadata(df: pd.DataFrame, columns: list[str], sort_by: list[str] | None = None) -> pd.DataFrame:
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[columns]
    if sort_by:
        df = df.sort_values(sort_by, kind='stable').reset_index(drop=True)
    return df


def build_output_paths(output_prefix: str | Path) -> tuple[Path, Path]:
    output_prefix = Path(output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix('.png')
    pdf_path = output_prefix.with_suffix('.pdf')
    return (png_path, pdf_path)


def write_df_to_sheet(df: pd.DataFrame, writer: pd.ExcelWriter, sheet_name: str) -> None:
    df.to_excel(writer, sheet_name=sheet_name, index=False)


def excel_sheets_to_csv(
    input_excel: str | Path,
    output_dir: str | Path,
    sheets: str | list[str] | None = None,
    encoding: str = 'utf-8-sig',
    output_names: str | dict[str, str] | None = None,
) -> list[Path]:
    """
    Convert one or more sheets in an Excel file to CSV files.

    - sheets=None: convert all sheets
    - sheets='Sheet1': convert one sheet
    - sheets=['A', 'B']: convert specific sheets
    - output_names:
      - str: when converting one sheet, use this filename
      - dict: map sheet_name -> filename
    """
    input_excel = Path(input_excel)
    output_dir = Path(output_dir)
    if not input_excel.exists():
        raise FileNotFoundError(f'Input Excel not found: {input_excel}')
    output_dir.mkdir(parents=True, exist_ok=True)
    xls = pd.ExcelFile(input_excel)
    all_sheets = [str(s) for s in xls.sheet_names]
    if sheets is None:
        target_sheets = all_sheets
    elif isinstance(sheets, str):
        target_sheets = [sheets]
    else:
        target_sheets = list(sheets)
    missing = [s for s in target_sheets if s not in all_sheets]
    if missing:
        raise ValueError(f'Sheet(s) not found: {missing}. Available sheets: {all_sheets}')
    written_files: list[Path] = []
    for sheet_name in target_sheets:
        df = pd.read_excel(input_excel, sheet_name=sheet_name)
        safe_sheet_name = ''.join((c if c not in '\\/:*?"<>|' else '_' for c in sheet_name))
        default_name = f'{input_excel.stem}__{safe_sheet_name}.csv'
        if output_names is None:
            filename = default_name
        elif isinstance(output_names, str):
            if len(target_sheets) != 1:
                raise ValueError('When output_names is a string, exactly one target sheet is required.')
            filename = output_names
        else:
            filename = output_names.get(sheet_name, default_name)
        if not filename.lower().endswith('.csv'):
            filename = f'{filename}.csv'
        csv_path = output_dir / filename
        df.to_csv(csv_path, index=False, encoding=encoding)
        written_files.append(csv_path)
    return written_files


def csv_to_excel_sheet(
    csv_file: str | Path,
    output_excel: str | Path,
    sheet_name: str,
    mode: ExcelWriteMode = 'a',
    if_sheet_exists: IfSheetExists = 'replace',
    encoding: str = 'utf-8-sig',
) -> Path:
    """
    Write one CSV file into a target Excel sheet.

    - mode='a' appends/updates workbook, mode='w' creates/overwrites workbook.
    - if_sheet_exists is used when mode='a' (e.g. 'replace', 'new', 'overlay').
    """
    csv_file = Path(csv_file)
    output_excel = Path(output_excel)
    if not csv_file.exists():
        raise FileNotFoundError(f'CSV not found: {csv_file}')
    df = pd.read_csv(csv_file, encoding=encoding)
    output_excel.parent.mkdir(parents=True, exist_ok=True)
    if mode not in {'a', 'w'}:
        raise ValueError("mode must be 'a' or 'w'")
    if mode == 'a' and output_excel.exists():
        with pd.ExcelWriter(output_excel, mode='a', if_sheet_exists=if_sheet_exists) as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        with pd.ExcelWriter(output_excel, mode='w') as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    return output_excel


__all__ = [
    'read_excel_data',
    'read_csv_data',
    'to_long',
    'metric_pivot',
    'finalize_plot_metadata',
    'build_output_paths',
    'write_df_to_sheet',
    'excel_sheets_to_csv',
    'csv_to_excel_sheet',
]
