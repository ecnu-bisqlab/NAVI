import random
import sys
from math import pi

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import (
    CDKMRippleCarryAdder,
    EfficientSU2,
    HGate,
    IntegerComparator,
    IQP,
    MCMT,
    Permutation,
    QFT,
    QuantumVolume,
    RZGate,
)
from qiskit.circuit.random import random_circuit


SUPPORTED_BENCHMARKS = (
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
)


def select_circuit(
    name,
    num_qubits,
    num_qpus,
    qpus,
    basis_gates,
    two_qubit_gates,
):
    if name not in SUPPORTED_BENCHMARKS:
        allowed = ", ".join(SUPPORTED_BENCHMARKS)
        raise ValueError(
            f"Unknown circuit name: {name!r}. Supported benchmarks: {allowed}."
        )

    assert num_qpus == len(qpus), "[ERROR] num_qpus != len(qpus)"

    optimization_level = 3
    if name == "QV":
        circ = QuantumVolume(num_qubits, 10, seed=26).decompose()
    elif name == "IQP":
        circ = iqp_circuit(num_qubits).decompose()
    elif name == "Random":
        circ = random_circuit(num_qubits, 10, max_operands=4, seed=42)
    elif name == "QFTAdder":
        circ = qft_adder(num_qubits)
    elif name == "PQC":
        circ = pqc(num_qubits).decompose()
    elif name == "Permutation":
        optimization_level = 0
        circ = Permutation(num_qubits, seed=21)
    elif name == "BV":
        circ = bv(num_qubits).decompose()
    elif name == "SwapTest":
        circ = swap_test(num_qubits).decompose()
    elif name == "MCMT":
        circ = mcmt_circuit(num_qubits).decompose()
    elif name == "VQE":
        circ = vqe(num_qubits)
    elif name == "Comparator":
        circ = comparator_circuit(num_qubits)
    else:
        circ = cuccaro_adder(num_qubits).decompose()

    if sum(qpus) < circ.num_qubits:
        print(
            f"[WARNING] sum(qpus) ({sum(qpus)}) < "
            f"circ.num_qubits ({circ.num_qubits})"
        )
        print("[WARNING] Reallocate QPU capacities.")
        qpu_capacity = circ.num_qubits // num_qpus + 1
        if qpu_capacity % 2 == 1:
            qpu_capacity += 1
        qpus = [qpu_capacity] * num_qpus

    trans_circ = transpile(
        circ,
        basis_gates=basis_gates,
        optimization_level=optimization_level,
        seed_transpiler=42,
    )

    gate_counts = trans_circ.count_ops()
    total_gates = sum(gate_counts.values())
    assert total_gates > 0, "[ERROR] An empty circuit."

    two_qubit_gate_counts = {
        gate: count
        for gate, count in gate_counts.items()
        if gate in two_qubit_gates
    }
    num_2q_gates = sum(two_qubit_gate_counts.values())

    messages = (
        f"[INFO] {name} #Qubits: {trans_circ.num_qubits}",
        f"[INFO] {name} Depth: {trans_circ.depth()}",
        f"[INFO] {name} #Gates: {total_gates}",
        f"[INFO] {name} #2Q Gates: {num_2q_gates}",
        f"[INFO] {num_qpus} QPUs: {qpus}\n",
    )
    for message in messages:
        print(message)
    for message in messages:
        print(message, file=sys.stderr)

    task_info = {
        "Circuit": name,
        "#Qubits": trans_circ.num_qubits,
        "#Gates": total_gates,
        "#2Q Gates": num_2q_gates,
        "Depth": trans_circ.depth(),
        "#QPUs": num_qpus,
        "QPUs": qpus,
        "Basis Gates": basis_gates,
        "2Q Gate Names": two_qubit_gates,
    }

    return circ, trans_circ, task_info


def cuccaro_adder(num_qubits):
    actual_qubits = num_qubits
    if actual_qubits < 4:
        raise ValueError("full adder requires at least 4 qubits")
    if actual_qubits % 2 != 0:
        actual_qubits -= 1
        print(
            "Warning: full adder requires an even number of qubits; "
            f"adjusted {num_qubits} to {actual_qubits}."
        )

    adder = CDKMRippleCarryAdder(
        num_state_qubits=(actual_qubits - 2) // 2,
        kind="full",
    )
    assert actual_qubits == adder.num_qubits
    return adder


def comparator_circuit(num_qubits):
    if num_qubits < 2:
        raise ValueError("num_qubits must be at least 2 for IntegerComparator.")

    for state_qubits in range(num_qubits, 0, -1):
        comparator = IntegerComparator(
            num_state_qubits=state_qubits,
            value=2 ** (state_qubits - 1),
            geq=True,
        )
        if comparator.num_qubits <= num_qubits:
            circuit = QuantumCircuit(
                comparator.num_qubits,
                name=f"Comparator{comparator.num_qubits}",
            )
            circuit.compose(
                comparator,
                qubits=range(comparator.num_qubits),
                inplace=True,
            )
            return circuit.decompose(reps=10)

    raise RuntimeError("Failed to construct a comparator circuit.")


def qft_adder(num_qubits):
    circuit = QuantumCircuit(num_qubits, name="Adder")
    circuit.h(0)
    circuit.h(1)

    register_size = num_qubits // 2
    qft = QFT(register_size, do_swaps=False)
    target_register = range(register_size, num_qubits)
    circuit.append(qft, target_register)

    for control in range(register_size):
        target = control + register_size
        for exponent, source in enumerate(range(control, register_size), start=1):
            circuit.cp(2 * pi / (2**exponent), source, target)

    circuit.append(qft.inverse(), target_register)
    return circuit.decompose()


def mcmt_circuit(num_qubits, gate_width=15, gate_count=10):
    if gate_width > num_qubits:
        raise ValueError("gate_width must be no larger than num_qubits.")

    circuit = QuantumCircuit(num_qubits)
    num_controls = gate_width // 2
    num_targets = gate_width - num_controls

    for _ in range(gate_count):
        base_gate = random.choice(
            [HGate(), RZGate(np.random.uniform(0, 2 * pi))]
        )
        gate = MCMT(base_gate, num_controls, num_targets)
        qargs = random.sample(range(num_qubits), gate_width)
        circuit.append(gate, qargs)

    return circuit


def iqp_circuit(num_qubits):
    print(f"[IQP] #Qubits: [{num_qubits}]")
    matrix = np.random.randint(0, 10, size=(num_qubits, num_qubits))
    symmetric_matrix = (matrix + matrix.T) // 2
    return IQP(symmetric_matrix)


def pqc(num_qubits, connections_per_qubit=5):
    circuit = QuantumCircuit(num_qubits)
    for control in range(num_qubits):
        candidates = [qubit for qubit in range(num_qubits) if qubit != control]
        targets = np.random.choice(
            candidates,
            size=min(connections_per_qubit, len(candidates)),
            replace=False,
        )
        for target in targets:
            circuit.crz(np.random.rand() * 2 * pi, control, target)
    return circuit


def swap_test(num_qubits, prep_depth=1, seed=26):
    if num_qubits < 3:
        raise ValueError("SwapTest requires at least 3 qubits.")
    if prep_depth < 1:
        raise ValueError("prep_depth must be a positive integer.")

    actual_qubits = num_qubits
    if actual_qubits % 2 == 0:
        actual_qubits -= 1
        print(
            "Warning: SwapTest requires 2n+1 qubits; "
            f"adjusted {num_qubits} to {actual_qubits}."
        )

    register_size = (actual_qubits - 1) // 2
    psi_qubits = list(range(1, 1 + register_size))
    phi_qubits = list(range(1 + register_size, actual_qubits))
    rng = np.random.default_rng(seed)
    circuit = QuantumCircuit(actual_qubits, name="SwapTest")

    for _ in range(prep_depth):
        for qubit in psi_qubits + phi_qubits:
            circuit.ry(float(rng.uniform(0, 2 * pi)), qubit)
            circuit.rz(float(rng.uniform(0, 2 * pi)), qubit)
        for register in (psi_qubits, phi_qubits):
            for left, right in zip(register[:-1], register[1:]):
                circuit.cx(left, right)

    circuit.h(0)
    for psi_qubit, phi_qubit in zip(psi_qubits, phi_qubits):
        circuit.cswap(0, psi_qubit, phi_qubit)
    circuit.h(0)
    return circuit


def bv(num_qubits, secret_bitstring=None, seed=26):
    data_qubits = num_qubits - 1
    if secret_bitstring is None:
        rng = np.random.default_rng(seed)
        secret_bitstring = "".join(
            rng.choice(["0", "1"], size=data_qubits)
        )

    secret_bitstring = secret_bitstring[:data_qubits].ljust(data_qubits, "0")
    circuit = QuantumCircuit(num_qubits)
    circuit.x(data_qubits)
    circuit.h(range(num_qubits))

    for qubit, bit in enumerate(secret_bitstring):
        if bit == "1":
            circuit.cx(qubit, data_qubits)

    circuit.h(range(data_qubits))
    return circuit


def vqe(num_qubits, reps=5):
    if num_qubits < 1:
        raise ValueError("num_qubits must be >= 1")
    if reps < 1:
        raise ValueError("reps must be >= 1")

    return EfficientSU2(
        num_qubits=num_qubits,
        reps=reps,
        entanglement="linear",
        flatten=True,
        name="VQE",
    )
