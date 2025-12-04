"""auto_k_mfvi.py
Utility routines to automatically choose the number of topics (K) for the
MFVI LFA model via cross-validation.

Typical usage
-------------
>>> from lfa._utils.auto_k_mfvi import select_k_mfvi
>>> best_k, results = select_k_mfvi(W_binary, candidate_Ks=[5, 10, 15])

The function is intentionally dependency-free (no sklearn) and works on any
binary NumPy array (subjects × diseases). It can therefore be called from
HPC or local scripts without relying on the CLI that exists in
``src/scripts/cv_select_k_mfvi.py``.
"""
from __future__ import annotations

import time
from typing import List, Tuple, Dict, Any

import numpy as np

from lfa._utils.mfvi_eval import infer_theta_single_row, compute_perplexity
from lfa._models.mfvi_model import MFVIModel
from lfa._utils.mfvi_monitor import ELBOMonitor

__all__ = ["select_k_mfvi"]


def _kfold_indices(n_samples: int, n_folds: int, seed: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Generate train / validation indices for each fold."""
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)

    fold_sizes = [n_samples // n_folds] * n_folds
    for i in range(n_samples % n_folds):
        fold_sizes[i] += 1

    folds = []
    start = 0
    for size in fold_sizes:
        val_idx = indices[start : start + size]
        train_idx = np.setdiff1d(indices, val_idx, assume_unique=True)
        folds.append((train_idx, val_idx))
        start += size
    return folds


def _train_mfvi(
    W: np.ndarray,
    alpha: np.ndarray,
    num_topics: int,
    max_iterations: int,
    convergence_threshold: float,
):
    """Train MFVI and return model (β, θ, etc.)."""
    model = MFVIModel(W, alpha, num_topics)
    monitor = ELBOMonitor(convergence_threshold, max_iterations)

    while True:
        elbo_before, elbo_after, param_changes = model.update_parameters()
        if monitor.check_convergence(elbo_before, elbo_after, param_changes):
            break
    return model


def _cv_score_single_K(
    W: np.ndarray,
    alpha: np.ndarray,
    K: int,
    n_folds: int,
    max_iterations: int,
    convergence_threshold: float,
    inference_max_iter: int,
    inference_tol: float,
    seed: int,
) -> Tuple[float, float, List[float], List[float]]:
    """Return mean perplexity & mean BIC together with per-fold values for candidate K."""
    fold_perps: List[float] = []
    fold_bics: List[float] = []
    folds = _kfold_indices(W.shape[0], n_folds=n_folds, seed=seed)

    D = W.shape[1]
    num_params = (K + 1) * D  # β parameters only

    for train_idx, val_idx in folds:
        W_train = W[train_idx]
        W_val = W[val_idx]

        # Train on training fold
        model = _train_mfvi(
            W_train,
            alpha=alpha,
            num_topics=K + 1,  # include healthy topic
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
        )
        beta_hat = model.beta  # (K+1, D)

        # Compute predictions for validation fold
        P_pred = np.zeros_like(W_val, dtype=float)
        for i, row in enumerate(W_val):
            theta_exp, _ = infer_theta_single_row(
                W_row=row,
                beta=beta_hat,
                alpha=alpha,
                max_iterations=inference_max_iter,
                convergence_threshold=inference_tol,
            )
            P_pred[i] = theta_exp @ beta_hat

        perp = compute_perplexity(W_val, P_pred)
        fold_perps.append(float(perp))

        # Convert perplexity to log-likelihood and then BIC
        N_val = W_val.size
        log_likelihood = -N_val * np.log(perp)
        bic = -2 * log_likelihood + num_params * np.log(N_val)
        fold_bics.append(float(bic))

    return float(np.mean(fold_perps)), float(np.mean(fold_bics)), fold_perps, fold_bics


def select_k_mfvi(
    W: np.ndarray,
    candidate_Ks: List[int],
    *,
    alpha: Optional[np.ndarray] = None,
    n_folds: int = 5,
    max_iterations: int = 2000,
    convergence_threshold: float = 1e-6,
    inference_max_iter: int = 100,
    inference_tol: float = 1e-4,
    seed: int = 0,
) -> Tuple[int, List[Dict[str, Any]]]:
    """Choose the best K (excluding healthy topic) for MFVI via CV.

    Parameters
    ----------
    W : np.ndarray
        Binary matrix (subjects × diseases).
    candidate_Ks : List[int]
        Candidate numbers of *non-healthy* topics to evaluate.
    alpha : np.ndarray, optional
        Dirichlet prior for each topic. If None uses 1/10 for each (K+1) topic.
    Other parameters control optimisation and CV.

    Returns
    -------
    best_k : int
        K value with the lowest mean perplexity.
    results : List[Dict]
        One dict per candidate with keys: "K", "mean_perplexity", "mean_bic", "fold_perplexities", "fold_bics".
    """
    if not ((W == 0) | (W == 1)).all():
        raise ValueError("Input matrix W must be binary (0/1).")

    results: List[Dict[str, Any]] = []
    t0 = time.time()

    for K in candidate_Ks:
        K_alpha = np.ones(K + 1) / 10 if alpha is None else alpha
        mean_perp, mean_bic, fold_perps, fold_bics = _cv_score_single_K(
            W,
            alpha=K_alpha,
            K=K,
            n_folds=n_folds,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            inference_max_iter=inference_max_iter,
            inference_tol=inference_tol,
            seed=seed,
        )
        results.append({
            "K": K,
            "mean_perplexity": mean_perp,
            "mean_bic": mean_bic,
            "fold_perplexities": fold_perps,
            "fold_bics": fold_bics,
        })

    best_entry = min(results, key=lambda d: d["mean_bic"])
    best_k = best_entry["K"]

    elapsed = time.time() - t0
    best_entry["elapsed_sec"] = elapsed

    return best_k, results
