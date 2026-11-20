# NAVI

NAVI is a noise-aware distributed quantum circuit compiler for multi-QPU systems. It partitions and maps quantum circuits across connected QPUs while considering device and network properties.

## Setup

NAVI requires Python 3.10.

```bash
pip install -r requirements.txt
cp config.example.json config.json
```

Update `device_properties_folder` in `config.json` to point to the directory containing the backend property files.

## SliceOpt DP/greedy-table comparison

NAVI uses the SliceOpt greedy-table strategy by default. Set
`compiler_config.navi.slice_strategy` to `greedy_table` or explicitly to `dp`
for legacy comparisons; both strategies reuse the same zero-cut partition
table, while `greedy_table` selects the longest feasible interval at each
boundary.

Validate the paired experiment configurations and print the commands without
starting the suite:

```bash
python scripts/run_slice_strategy_comparison.py --validate-only
```

Run the paired full-NAVI 80-qubit, 4x20, mesh-grid suite. Both configurations
disable ablation mode and checkpoint evaluations, leave all NAVI/MapOpt/CommOpt
parameters at their defaults, and use the historical per-benchmark circuit
seeds. The network fidelity sampler uses seed 42 so the DP and greedy runs see
the same reproducible mesh:

```bash
python scripts/run_slice_strategy_comparison.py
```

The runner resumes verified completed circuits, refuses to append over partial
results, writes the paired raw summary to
`outputs/slice_strategy_comparison/summary.csv`, and writes greedy-minus-DP
deltas and timing speedups to
`outputs/slice_strategy_comparison/comparison.csv`.

## Run

```bash
python main.py -cname BV -nq 80 -core 4 -cap 20 -net mesh_grid
```

The example compiles an 80-qubit Bernstein-Vazirani circuit for four QPUs, each with a core capacity of 20 qubits, connected using a mesh-grid network.

- `-cname`: circuit name
- `-nq`: number of circuit qubits
- `-core`: number of QPUs
- `-cap`: core capacity of each QPU
- `-net`: inter-QPU network topology

## Naive ablation baseline

Evaluate the original 80-qubit, 4x20 chain ablation circuits with a fixed
contiguous partition and TeleGate execution, reusing the pinned NAVI checkpoint
circuits:

```bash
python scripts/run_naive_ablation.py
```

The runner validates all checkpoint circuits before evaluation and resumes from
completed rows in `outputs/navi_naive_ablation/naive_results.csv`. It preserves
`outputs/navi_ablation.csv` and writes the three-metric merged table to
`outputs/navi_ablation_with_naive.csv`. Use `--validate-only` to check the pinned
checkpoint provenance without evaluating the circuits.

## CommOpt parameter analysis

Run the original eleven non-Random ablation benchmarks at 80 requested qubits,
4x20 all-to-all QPUs, circuit seed 42, `W={1,2,4}`, and
`L_max={5,10,15}` with:

```bash
./run_commopt_parameter_analysis.sh
```

The optional positional argument is an experiment ID. Reusing an existing ID
is rejected; using a new ID preserves the run as another replicate:

```bash
./run_commopt_parameter_analysis.sh repeat_02
```

The runner also accepts explicit scale and benchmark options. For the complete
12-benchmark 200-qubit sweep on four 50-qubit QPUs, use:

```bash
NAVI_PYTHON=/path/to/python \
./run_commopt_parameter_analysis.sh \
  --experiment-id commopt_4x50_200q \
  --qubits 200 \
  --cores 4 \
  --capacity 50 \
  --network all_to_all \
  --benchmarks all
```

`--benchmarks` also accepts a comma-separated subset such as
`Random,IQP,QFTAdder`. `--widths` and `--spans` restrict the parameter grid;
for example, `--widths 2,4 --spans 5` runs only `(2,5)` and `(4,5)`.
`--skip-combinations` removes individual `W:L` pairs from that grid. The
interrupted 4x50 QFTAdder sweep can continue in a new batch without rerunning
its completed `W1_L5` point with:

```bash
NAVI_PYTHON=/path/to/python \
./run_commopt_parameter_analysis.sh \
  --experiment-id commopt_4x50_qft_retry_20260814 \
  --qubits 200 --cores 4 --capacity 50 --network all_to_all \
  --benchmarks QFTAdder \
  --widths 1,2,4 --spans 5,10,15 \
  --skip-combinations 1:5
```

Use `--help` to list all runner options.

If the project environment is not the active shell environment, select its
interpreter explicitly with `NAVI_PYTHON=/path/to/python`.

Raw runs are stored below `outputs/navi_param_runs/<experiment_id>/`. After
every completed `W,L_max` run, the script combines all historical runs with the
legacy Random results and atomically rebuilds the authoritative dataset at
`outputs/navi_param.csv`. It is therefore safe to interrupt a sweep and inspect
the completed rows. `is_valid` describes one run; `group_complete` says whether
that benchmark has all nine parameter combinations. The `requested_qubit_count`,
`qubit_count`, and `qpu_layout` columns distinguish 4x20 and 4x50 experiments.
Progress can be checked while the sweep is running with:

```bash
python -c "import pandas as pd; d=pd.read_csv('outputs/navi_param.csv'); print(d.groupby(['qpu_layout','benchmark']).size())"
```

Every master-dataset refresh also atomically rebuilds
`outputs/navi_param_4x50.csv`. This derived table keeps the earliest valid row
for each parameter point from 4x50 `IQP`, `QFTAdder`, `PQC`, `MCMT`, and `VQE`
runs that satisfy the atomic-subcircuit threshold, even when points come from
different experiment IDs. The master CSV continues to retain ineligible runs,
invalid runs, and later replicates for auditing.

The parameter-analysis configuration requires at least 15 atomic zero-cut
subcircuits. NAVI checks this immediately after zero-cut slicing; when a
benchmark has fewer blocks, the runner records its observed count and stops the
remaining parameter combinations for that benchmark before CommOpt starts.

To regenerate the CSV without rerunning NAVI, use:

```bash
python scripts/analyze_commopt_parameters.py \
  outputs/navi_param_runs outputs/commopt_parameter_analysis \
  --output-csv outputs/navi_param.csv
```

### VQE sensitivity without the full-span safeguard

The dedicated wrapper below fixes the VQE circuit and compiler settings,
disables the CommOpt full-span safeguard and TeleGate cache, and varies only
the all-to-all network-fidelity seed. It does not run unless invoked explicitly.
Run one seed first with:

```bash
NAVI_PYTHON=/path/to/python \
./run_vqe_commopt_sensitivity.sh \
  --seeds 42 \
  --experiment-prefix vqe_nofs
```

Use a new-seed list with the same prefix to add network instances without
overwriting the first batch:

```bash
NAVI_PYTHON=/path/to/python \
./run_vqe_commopt_sensitivity.sh \
  --seeds 43,44 \
  --experiment-prefix vqe_nofs
```

After each invocation, the wrapper summarizes all completed batches matching
the prefix below `outputs/vqe_commopt_sensitivity/<prefix>/`. A single seed has
no estimable sample standard deviation, so its SD cells remain empty. With
multiple seeds, summaries report the sample SD across network instances.

To run all five sensitivity benchmarks sequentially with every seed for the
fastest benchmark completed first, use:

```bash
NAVI_PYTHON=/path/to/python \
./run_five_commopt_sensitivity.sh
```

The fixed order is `VQE`, `MCMT`, `PQC`, `QFTAdder`, and `IQP`; within each
benchmark the network seeds run as 42, 43, and 44. Use `--dry-run` to print the
15 jobs without launching them. Complete jobs are skipped on a later
invocation, while incomplete experiment directories are never overwritten.

### Short-circuit CommOpt saturation check

The targeted short-circuit wrapper fixes `W=1` and tests whether increasing
`L_max` from 5 to 10 is useful. `SwapTest`, `Comparator`, `CuccaroAdder`, and
`QV` use `L_max={5,10}`; their `K<=10` makes 15 structurally identical to 10.
`Random` uses `L_max={5,10,15}`, where 15 has an effective full span of 12.
Each point is repeated with network seeds 42, 43, and 44:

```bash
NAVI_PYTHON=/path/to/python \
./run_short_commopt_saturation.sh \
  --experiment-prefix commopt_short_nofs
```

Use `--dry-run` to inspect the 15 benchmark/seed batches (33 parameter points)
without launching them. The wrapper disables the full-span safeguard and
TeleGate cache, reuses one greedy SliceOpt/max-match prefix per batch, and
writes audited raw, summary, and pairwise-comparison CSV files below
`outputs/short_commopt_saturation/<experiment_prefix>/`.

## S-OEE iteration-cap analysis

The standalone convergence experiment checks whether the Static OEE iteration
cap of 50 is sufficient for a selected benchmark circuit on four 50-qubit QPUs
with circuit seed 42. It uses the same circuit generation, transpilation,
interaction graph, initial partition, and OEE implementation as the compiler,
without running the full evaluator or writing mapping-record JSON files.

```bash
python scripts/analyze_oee_iterations.py
```

If the project environment is not active, invoke the script with its Python
interpreter directly:

```bash
/path/to/python scripts/analyze_oee_iterations.py
```

By default, the experiment scans consecutive iteration caps until two adjacent
caps produce the same normalized partition, then explicitly compares cap 50
with a reference cap of 100. Results are written to
`outputs/oee_iteration_analysis/random_200q_4x50_seed42/` as a trace CSV, a
summary JSON, and PNG/PDF convergence figures. Run `--help` to override the
circuit, circuit size, QPU layout, seeds, backend, calibration date, or caps.
For example, use `--benchmark QFTAdder`; when `--output-dir` is omitted, each
benchmark and layout receives its own output directory automatically. The
`--benchmark` choices cover the same 12 circuits used by the main experiment.

After running all 12 choices with the same layout, build the cross-benchmark
table and figure with:

```bash
python scripts/summarize_oee_iteration_suite.py
```

The plotted weighted cut is the weighted number of two-qubit interactions that
cross QPU partitions. For this all-to-all Static OEE setup it is the direct
communication/EPR proxy optimized by OEE. The resulting sufficiency verdict is
specific to this fixed circuit instance and must not be interpreted as a
worst-case guarantee for every random circuit.

## WBCP segment-count analysis

WBCP's legacy rule uses `depth // 20` as a window length before clamping that
length to `[30, 500]`; it does not always create exactly 20 segments. The
standalone sensitivity experiment retains that legacy result and compares it
with explicit target segment counts. Explicit targets use
`max(30, min(ceil(depth / target), 500))`, so every tested window remains within
the same bounds:

```bash
python scripts/analyze_wbcp_segments.py
```

The default Random-200/4x50 sweep evaluates target counts 5, 10, 15, 20, 25,
and 30, and writes CSV/JSON plus PNG/PDF figures below
`outputs/wbcp_segment_analysis/`. Passing `num_segments` is opt-in; normal WBCP
runs continue to use the unchanged legacy window rule.

To scan the communication/EPR objective efficiently, use communication mode.
It replays the evaluator's complete remote-operation state machine but skips
local routing and local-gate fidelity accumulation, neither of which changes
the EPR count:

```bash
python scripts/analyze_wbcp_segments.py \
  --benchmark Random \
  --evaluation-mode communication
```

Run this command for each of the 12 benchmark choices, then aggregate the
completed sweeps with:

```bash
python scripts/summarize_wbcp_segment_suite.py
```

The bounded all-to-all suite CSV/JSON and normalized-EPR heatmap are written
below
`outputs/wbcp_segment_analysis/suite_200q_4x50_seed42_all_to_all_bounded_communication/`. The
primary selection criterion is minimum EPR pairs; quality ties are retained,
then remote effective fidelity and runtime are used only to recommend one of
the tied counts. Both requested segment count and actual window count are
reported because short circuits can produce fewer windows than requested.
