"""
AQS Reference Implementation (skeleton)
- Provides functions to estimate the three AQS components and a streaming estimator.
- Includes clear placeholders for Qiskit/JAX device-specific calls.
- Includes a simple synthetic test runnable without Qiskit/JAX.

To use on real hardware:
- Replace `get_density_matrix(t)` with a Qiskit state/density extraction (or shadows estimator).
- Replace `get_gate_graph_laplacian(t)` with your gate-layer adjacency -> Laplacian builder.
- Replace `get_parameter_gradients(t)` with JAX-computed parameter-shift gradients for the ansatz.

Written to /mnt/data/aqs_reference_impl.py by ChatGPT.
"""

import numpy as np
from scipy.sparse import csgraph
from scipy.linalg import eigh

# ----------------------------- Core estimators -----------------------------

def purity_from_density_matrix(rho):
    """
    Compute purity Tr[rho^2]. rho is a square numpy array (density matrix).
    """
    rho = np.asarray(rho, dtype=np.complex128)
    assert rho.shape[0] == rho.shape[1], "rho must be square"
    # For numerical stability, ensure Hermitian
    rho = 0.5*(rho + rho.conj().T)
    return float(np.real_if_close(np.trace(rho.dot(rho))))

def algebraic_connectivity_from_laplacian(L):
    """
    Compute the Fiedler value (lambda_2) of Laplacian L.
    L can be a dense numpy array or scipy sparse matrix.
    """
    # Use eigh for symmetric matrices; compute smallest nonzero eigenvalue
    L = np.asarray(L, dtype=float)
    # compute eigenvalues in ascending order
    vals, vecs = eigh(L)
    # eigenvalues might be tiny negative due to numerical error; clip
    vals = np.clip(vals, 0.0, None)
    if vals.shape[0] < 2:
        return 0.0
    # Fiedler value is second smallest (index 1)
    return float(vals[1])

def trace_fisher_from_gradients(gradients):
    """
    Given gradients of log-likelihoods or parameter-shift gradients, estimate trace of Fisher information.
    gradients: array of shape (num_samples, num_params) representing d/dtheta log p(x|theta) samples
    For quantum parameter-shift gradients (deterministic per measurement batch), you may pass gradient vectors.
    Returns trace(F) ≈ sum_k Var(grad_k) if using classical Fisher approximation.
    """
    g = np.asarray(gradients, dtype=float)
    if g.ndim == 1:
        # single gradient vector -> approximate using outer product
        return float(np.dot(g, g))
    # empirical covariance across samples
    cov = np.cov(g, rowvar=False, bias=True)
    return float(np.trace(cov))

# ----------------------------- Normalization helpers -----------------------------

def normalize_purity(purity, n_qubits):
    """
    Map purity in [1/D, 1] to [0,1]. D = 2^n_qubits
    """
    D = 2**n_qubits
    min_p = 1.0 / D
    return float((purity - min_p) / (1.0 - min_p))

def normalize_connectivity(lambda2, n_active):
    """
    Normalize algebraic connectivity by the Fiedler value of the complete graph: n_active
    """
    if n_active <= 1:
        return 0.0
    return float(lambda2 / float(n_active))

def normalize_fisher_trace(trF_over_d, Fmax_est=1.0):
    """
    Normalize sqrt(TrF/d) by an estimated Fmax.
    trF_over_d: scalar value sqrt(TrF/d)
    Fmax_est: chosen normalization constant (user must set or estimate)
    """
    if Fmax_est <= 0:
        raise ValueError("Fmax_est must be positive")
    return float(trF_over_d / np.sqrt(Fmax_est))

# ----------------------------- AQS streaming estimator -----------------------------

class AQSStream:
    def __init__(self, n_qubits, n_active, d_params,
                 Fmax_est=1.0, alpha=0.05):
        """
        n_qubits: total qubits (for purity normalization)
        n_active: active nodes in gate graph (for connectivity normalization)
        d_params: ansatz parameter count (for Fisher trace normalization)
        Fmax_est: normalization constant for Fisher trace (choose principled bound)
        alpha: streaming update weight (exponential smoothing)
        """
        self.n_qubits = int(n_qubits)
        self.n_active = int(n_active)
        self.d_params = int(d_params)
        self.Fmax_est = float(Fmax_est)
        self.alpha = float(alpha)
        self.value = None  # running AQS value

    def update(self, rho=None, laplacian=None, gradients=None, trF=None):
        """
        Update streaming AQS given available diagnostics. The function accepts either:
        - gradients: array used to compute TrF via trace_fisher_from_gradients, OR
        - trF: direct scalar of TrF (preferred if available).
        rho: density matrix (numpy array)
        laplacian: Laplacian matrix for active gate graph (numpy array or sparse)
        """
        # purity
        if rho is None:
            raise ValueError("rho (density matrix) must be provided for purity estimation in this skeleton")
        purity = purity_from_density_matrix(rho)
        phi_norm = normalize_purity(purity, self.n_qubits)

        # connectivity
        if laplacian is None:
            raise ValueError("laplacian must be provided for connectivity")
        lambda2 = algebraic_connectivity_from_laplacian(laplacian)
        g_norm = normalize_connectivity(lambda2, self.n_active)

        # fisher trace
        if trF is None:
            if gradients is None:
                raise ValueError("Either gradients or trF must be provided for Fisher trace estimation")
            trF_est = trace_fisher_from_gradients(gradients)
        else:
            trF_est = float(trF)

        # compute sqrt(TrF / d)
        s_term = np.sqrt(max(trF_est, 0.0) / max(1, self.d_params))
        s_norm = normalize_fisher_trace(s_term, self.Fmax_est)

        # multiplicative AQS instant
        a_instant = phi_norm * g_norm * s_norm

        # update running value (exponential smoothing)
        if self.value is None:
            self.value = a_instant
        else:
            self.value = (1.0 - self.alpha) * self.value + self.alpha * a_instant

        return {
            'a_instant': float(a_instant),
            'aqs_stream': float(self.value),
            'phi_norm': float(phi_norm),
            'g_norm': float(g_norm),
            's_norm': float(s_norm),
            'lambda2': float(lambda2),
            'purity': float(purity),
            'trF_est': float(trF_est)
        }

# ----------------------------- Placeholders for Qiskit/JAX integration -----------------------------

def get_density_matrix_qiskit_placeholder(t):
    """
    Placeholder: replace with Qiskit code to extract the device's density matrix or classical shadow estimator.
    Example (pseudocode):
        job = execute(circuit_at_t, backend=backend, shots=shots)
        # use tomography or classical shadows to estimate rho
        rho = estimate_density_matrix_from_shadows(job.result())
    Return a numpy array density matrix.
    """
    raise NotImplementedError("Replace get_density_matrix_qiskit_placeholder with real Qiskit code.")

def get_gate_graph_laplacian_placeholder(t):
    """
    Placeholder: compute Laplacian of active gate graph at time t.
    The adjacency matrix should reflect active two-qubit gates in the layer (weights optional).
    """
    raise NotImplementedError("Replace get_gate_graph_laplacian_placeholder with real gate parsing code.")

def get_gradients_jax_placeholder(t):
    """
    Placeholder: compute parameter gradients via JAX (or parameter-shift via Qiskit if using finite-shot).
    Return a matrix of gradient samples or gradient vectors.
    """
    raise NotImplementedError("Replace get_gradients_jax_placeholder with device/training gradients.")

# ----------------------------- Synthetic test (no external deps) -----------------------------

def synthetic_density_matrix(n_qubits, purity_target=0.9):
    """
    Build a random density matrix with approximately the given purity.
    Method: mix a pure random state with maximally mixed state.
    """
    D = 2**n_qubits
    psi = np.random.randn(D) + 1j*np.random.randn(D)
    psi /= np.linalg.norm(psi)
    rho_pure = np.outer(psi, psi.conj())
    rho = purity_target * rho_pure + (1.0 - purity_target) * np.eye(D) / D
    return rho

def synthetic_laplacian(n_active, connectivity=0.3):
    """
    Build a random adjacency for n_active nodes with expected edge density connectivity and return Laplacian.
    """
    A = np.random.rand(n_active, n_active) < connectivity
    A = np.triu(A, 1).astype(float)
    A = A + A.T
    np.fill_diagonal(A, 0.0)
    L = np.diag(A.sum(axis=1)) - A
    return L

def synthetic_gradients(num_samples, d_params, scale=1.0):
    """
    Create synthetic gradient samples with controllable variance.
    """
    return np.random.randn(num_samples, d_params) * scale

def run_synthetic_demo():
    print("Running synthetic AQS demo...")
    n_qubits = 4
    n_active = 4
    d_params = 10
    stream = AQSStream(n_qubits=n_qubits, n_active=n_active, d_params=d_params,
                       Fmax_est=5.0, alpha=0.15)

    T = 50
    results = []
    for t in range(T):
        # simulate drifting purity and connectivity and fisher magnitude
        purity_t = 0.7 + 0.3 * np.sin(0.1 * t) * np.random.rand()
        rho = synthetic_density_matrix(n_qubits, purity_target=min(max(purity_t, 1.0/(2**n_qubits)), 1.0))
        L = synthetic_laplacian(n_active, connectivity=0.2 + 0.4*np.abs(np.sin(0.05*t)))
        grads = synthetic_gradients(num_samples=20, d_params=d_params, scale=0.5 + 0.5*np.abs(np.cos(0.07*t)))
        out = stream.update(rho=rho, laplacian=L, gradients=grads)
        out['t'] = t
        results.append(out)

    # print summary
    for x in results[::max(1, T//10)]:
        print(f"t={x['t']:2d}  A_instant={x['a_instant']:.4f}  A_stream={x['aqs_stream']:.4f}  purity={x['purity']:.4f}  lambda2={x['lambda2']:.4f}  trF={x['trF_est']:.4f}")
    return results

if __name__ == "__main__":
    run_synthetic_demo()
