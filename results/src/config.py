from dataclasses import dataclass, field


@dataclass
class PlotConfig:
    required_meta: list[str] = field(default_factory=lambda: [
        'Circuit', '#Qubits', 
        '#Gates', '#2Q Gates', '#Depth', 
        '#QPUs', 'QPUs', 'net_type', 'Metrics',
    ])
    method_cols: list[str] = field(default_factory=lambda: ['S-OEE', 'FGP-rOEE', 'AutoComm', 'WBCP', 'NAVI (Ours)'])
    method_rename: dict[str, str] = field(default_factory=dict)
    circuit_rename: dict[str, str] = field(default_factory=dict)
    method_order: list[str] = field(default_factory=lambda: ['S-OEE', 'FGP-rOEE', 'AutoComm', 'WBCP', 'NAVI (Ours)'])
    method_color: dict[str, str] = field(default_factory=lambda: {
        'S-OEE': '#737373',
        'FGP-rOEE': '#3E608D',
        'AutoComm': '#8CBF87',
        'WBCP': '#F2B07E',
        'NAVI (Ours)': '#d62728',
    })
    method_marker: dict[str, str] = field(default_factory=lambda: {
        'S-OEE': 'o',
        'FGP-rOEE': 's',
        'WBCP': '^',
        'AutoComm': 'D',
        'NAVI (Ours)': '*',
    })
    method_hatch: dict[str, str] = field(default_factory=lambda: {
        'S-OEE': '',
        'FGP-rOEE': '///',
        'WBCP': '\\\\\\\\',
        'AutoComm': 'xx',
        'NAVI (Ours)': '.',
    })
    bench_order: list[str] = field(default_factory=lambda: [
        'QV', 'IQP', 'Random',
        'QFTAdder', 'PQC', 'Permutation',
        'BV', 'SwapTest', 'MCMT',
        'VQE', 'Comparator', 'CuccaroAdder',
    ])
    baseline_order: list[str] | None = None
    anchor_method: str = 'NAVI (Ours)'
    colors: list[str] = field(default_factory=lambda: ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2', '#937860'])
    markers: list[str] = field(default_factory=lambda: ['o', 's', '^', 'D', 'P', 'X'])
    hatches: list[str] = field(default_factory=lambda: ['', '///', '\\\\\\', 'xx', '..', '++'])
    figsize: tuple[float, float] = (16, 6)
    pub_rc_params: dict[str, object] = field(default_factory=lambda: {
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        'font.size': 18,
        'axes.labelsize': 18,
        'axes.titlesize': 19,
        'xtick.labelsize': 18,
        'ytick.labelsize': 18,
        'legend.fontsize': 18,
        'figure.titlesize': 19,
        'font.weight': 'bold',
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'axes.linewidth': 1.2,
        'lines.linewidth': 1.5,
        'hatch.linewidth': 1.0,
        'hatch.color': '#969696',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'savefig.dpi': 300,
    })

DEFAULT_CONFIG = PlotConfig()

__all__ = ['PlotConfig', 'DEFAULT_CONFIG']
