import numpy as np
from scipy.special import digamma

EPS = 1e-100


def _update_e_z_single_row(beta: np.ndarray, W_row: np.ndarray, e_log_theta: np.ndarray) -> np.ndarray:
    """Update expected assignment probabilities z for one subject.

    Args:
        beta: (K, D) topic-disease probabilities.
        W_row: (D,) binary disease vector for a single subject.
        e_log_theta: (K,) expected log theta for this subject.

    Returns:
        z: (D, K) updated topic assignment probabilities for this subject.
    """
    K, D = beta.shape
    z = np.zeros((D, K))
    log_rho = np.log(beta + EPS)
    log_one_minus_rho = np.log(1 - beta + EPS)

    for d in range(D):
        log_phi = e_log_theta + W_row[d] * log_rho[:, d] + (1 - W_row[d]) * log_one_minus_rho[:, d]
        log_phi -= np.max(log_phi)  # numerical stability
        phi = np.exp(log_phi)
        z[d] = phi / np.sum(phi)
    return z


def infer_theta_single_row(
    W_row: np.ndarray,
    beta: np.ndarray,
    alpha: np.ndarray,
    max_iterations: int = 200,
    convergence_threshold: float = 1e-6,
):
    """Perform MFVI updates for a single subject given fixed beta.

    Only theta and z are updated. Beta matrix remains fixed.

    Args:
        W_row: (D,) binary vector for diseases of one subject.
        beta: (K, D) learned topic-disease matrix.
        alpha: (K,) Dirichlet hyper-parameter (same used in training).
        max_iterations: maximum coordinate-ascent steps.
        convergence_threshold: stop when mean absolute change in z < threshold.

    Returns:
        theta_exp: (K,) expectation of theta (Dirichlet mean).
        z: (D, K) probabilistic topic assignments.
    """
    K, D = beta.shape
    # Initialise z uniformly
    z = np.ones((D, K)) / K

    for _ in range(max_iterations):
        prev_z = z.copy()

        # Update E[theta] parameters and E[log theta]
        e_theta = alpha + np.sum(z, axis=0)  # shape (K,)
        e_log_theta = digamma(e_theta) - digamma(np.sum(e_theta))

        # Update z
        z = _update_e_z_single_row(beta, W_row, e_log_theta)

        # Check convergence
        mean_change = np.mean(np.abs(z - prev_z))
        if mean_change < convergence_threshold:
            break

    theta_exp = e_theta / np.sum(e_theta)
    return theta_exp, z


def compute_perplexity(W_true: np.ndarray, P_pred: np.ndarray) -> float:
    """Compute perplexity for binary data.

      perplexity = exp( -LL / N ) where N = total entries.
    """
    assert W_true.shape == P_pred.shape, "Shapes of true and predicted matrices must match"

    # Clip to avoid log(0)
    P_pred = np.clip(P_pred, EPS, 1 - EPS)
    LL = np.sum(W_true * np.log(P_pred) + (1 - W_true) * np.log(1 - P_pred))
    N = np.prod(W_true.shape)
    perplexity = np.exp(-LL / N)
    return perplexity
