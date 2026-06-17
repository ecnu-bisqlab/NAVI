from .style import set_pub_style
from .io import (
    read_excel_data,
    read_csv_data,
    to_long,
    metric_pivot,
    finalize_plot_metadata,
    build_output_paths,
    write_df_to_sheet,
    excel_sheets_to_csv,
    csv_to_excel_sheet,
)
from .plot_benchmarks import plot_fnorm, plot_pareto_scatter
from .plot_scalability import plot_dual_axis_epairs_fnorm, plot_dual_axis_epairs_error
from .plot_networks import plot_network_robustness
from .pdf_tools import merge_pdfs_by_filename, merge_pdfs_in_order
from .config import PlotConfig, DEFAULT_CONFIG

__all__ = [
    'set_pub_style',
    'read_excel_data', 'read_csv_data', 'to_long', 'build_output_paths', 'write_df_to_sheet',
    'excel_sheets_to_csv', 'csv_to_excel_sheet',
    'metric_pivot', 'finalize_plot_metadata',
    'plot_fnorm', 'plot_pareto_scatter',
    'plot_dual_axis_epairs_fnorm', 'plot_dual_axis_epairs_error',
    'plot_networks',
    'merge_pdfs_by_filename', 'merge_pdfs_in_order',
    'PlotConfig', 'DEFAULT_CONFIG',
    'analyze_sheet', 'analyze_workbook',
]


def __getattr__(name):
    if name in {'analyze_sheet', 'analyze_workbook'}:
        from .analyze import analyze_sheet, analyze_workbook

        return {
            'analyze_sheet': analyze_sheet,
            'analyze_workbook': analyze_workbook,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
