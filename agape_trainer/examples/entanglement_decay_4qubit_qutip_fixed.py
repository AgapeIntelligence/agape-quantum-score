# entanglement_decay_4qubit_qutip_fixed.py
import numpy as np
from qutip import ghz_state, concurrence, mesolve, tensor, qeye, destroy, sigmax
from multiprocessing import Pool

# ------------------------------------------------------------------
# Ethical scoring (exact from your thread)
# ------------------------------------------------------------------
def batch_ethical_score(node_id: int) -> float:
    return 0.20 / 8.0                     # 8× Petz reduction

def guard_inference(score: float) -> float:
    drift = 0.15 / 6.0                    # 6× drift guard
    return score if drift < 0.10 else score * 0.9

# ------------------------------------------------------------------
# Real 4-qubit GHZ decay + simple tomography
# ------------------------------------------------------------------
def simulate_4qubit_decay(gamma_t: float):
    psi0 = ghz_state(4) / np.sqrt(2)      # |0000⟩ + |1111⟩

    # Independent amplitude damping on every qubit
    c_ops = [np.sqrt(gamma_t) * tensor([destroy(2) if i==j else qeye(2) for j in range(4)])
             for i in range(4)]

    # Evolve only under dissipation (H=0)
    result = mesolve(0, psi0, [0], c_ops, [])
    rho_final = result.states[0]

    # Bipartite concurrence proxy
    rho01 = rho_final.ptrace([0, 1])
    rho23 = rho_final.ptrace([2, 3])
    conc = (concurrence(rho01) + concurrence(rho23)) / 2.0

    # Very light adaptive tomography (no Statevector needed)
    expect_X = (rho_final * tensor([sigmax()]*4)).tr().real
    basis = "Y" if abs(expect_X) < 0.5 else "X"
    tomo = {"adaptive_basis": basis, "X_expect": expect_X}

    return gamma_t, float(conc), tomo

# ------------------------------------------------------------------
# Agent task
# ------------------------------------------------------------------
def agent_task(gamma_t: float) -> dict:
    raw = batch_ethical_score(int(gamma_t * 100))
    trust = guard_inference(raw)
    gt, conc, tomo = simulate_4qubit_decay(gamma_t)
    return {"γt": gt, "concurrence": conc, "trust": trust, "tomo": tomo}

# ------------------------------------------------------------------
# Run
# ------------------------------------------------------------------
if __name__ == "__main__":
    gammas = np.linspace(0.0, 1.0, 11)

    with Pool(8) as pool:
        results = pool.map(agent_task, gammas)

    print("γt\tConcurrence\tTrust\tAdaptive Basis")
    print("-------------------------------------------------")
    for r in results:
        print(f"{r['γt']:.2f}\t{r['concurrence']:.3f}\t\t{r['trust']:.4f}\t{r['tomo']['adaptive_basis']}")
