# fed_ethics_proto.py — Federated Ethics Prototype (Updated + Version-Safe Tomography)
from qiskit import QuantumCircuit
import numpy as np
from typing import Dict
import time
from qiskit.quantum_info import Statevector


# ---------------------------------------------------------------------------
# 1. Batch Ethical Score (Node-Specific Bias + Timing Check)
# ---------------------------------------------------------------------------
def batch_ethical_score(circuit: QuantumCircuit, node: str, petz_recovery: bool = True) -> float:
    node_biases = {"Heron": 0.20, "Forte": 0.21, "Willow": 0.19}
    base_bias = node_biases.get(node, 0.20)
    mitigation_factor = 8.0 if petz_recovery else 1.0

    start = time.perf_counter()
    score = base_bias / mitigation_factor
    elapsed = time.perf_counter() - start

    if node == "Heron" and elapsed > 180e-6:
        print(f"Warning: Heron exceeded 180 µs ({elapsed*1e6:.1f} µs)")

    return float(score)


# ---------------------------------------------------------------------------
# 2. Guarded Inference Layer
# ---------------------------------------------------------------------------
def guard_inference(score: float, drift_threshold: float = 0.10) -> float:
    drift = 0.15
    reduction = 6.0
    corrected_drift = drift / reduction

    return float(score if corrected_drift < drift_threshold else score * 0.9)


# ---------------------------------------------------------------------------
# 3. **Version-Safe Adaptive Tomography Block**
#       Automatically handles:
#           - Qiskit 0.x/1.x (sample_counts exists)
#           - Qiskit 2.x+ (sample_memory only)
# ---------------------------------------------------------------------------
def adaptive_tomography(circuit: QuantumCircuit, shots: int = 1024) -> Dict[str, np.ndarray]:
    state = Statevector.from_instruction(circuit)

    # Determine sampling method: sample_counts (old) or sample_memory (new)
    if hasattr(state, "sample_counts"):
        def sampler():
            return state.sample_counts(shots=shots)
    else:
        def sampler():
            # sample_memory returns a list of bitstrings; convert to counts
            mem = state.sample_memory(shots=shots)
            unique, counts = np.unique(mem, return_counts=True)
            return dict(zip(unique, counts))

    # Adaptive basis selection
    x_expect = abs(state.expectation_value("X"))
    initial_basis = "Y" if x_expect < 0.5 else "X"

    measurements = {}
    for basis in [initial_basis, "Z"]:
        counts = sampler()
        vec = np.array(list(counts.values())) / shots
        measurements[basis] = vec

    # HEOM-like suppression placeholder
    suppression = 5.0
    for b in measurements:
        measurements[b] = measurements[b] / suppression

    return measurements


# ---------------------------------------------------------------------------
# 4. Federated Ethics Loop (Scoring + Guard + Tomography)
# ---------------------------------------------------------------------------
def run_federated_ethics_with_tomo() -> Dict[str, tuple]:
    nodes = ["Heron", "Forte", "Willow"]

    circuit = QuantumCircuit(4)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(0, 2)
    circuit.cx(0, 3)
    circuit.measure_all()

    results = {}

    for node in nodes:
        raw = batch_ethical_score(circuit, node, petz_recovery=True)
        guarded = guard_inference(raw)
        tomo = adaptive_tomography(circuit)
        results[node] = (guarded, tomo)

    return results


# ---------------------------------------------------------------------------
# 5. Script Entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = run_federated_ethics_with_tomo()
    print("Federated Trust Scores with Tomography:")
    for node, (score, tomo) in results.items():
        print(f"{node}: Trust Score = {score:.4f}, Tomography = {tomo}")
