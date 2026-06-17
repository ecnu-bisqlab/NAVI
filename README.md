# NAVI

NAVI is a noise-aware distributed quantum circuit compiler for multi-QPU systems. It partitions and maps quantum circuits across connected QPUs while considering device and network properties.

## Setup

NAVI requires Python 3.10.

```bash
pip install -r requirements.txt
cp config.example.json config.json
```

Update `device_properties_folder` in `config.json` to point to the directory containing the backend property files.

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
