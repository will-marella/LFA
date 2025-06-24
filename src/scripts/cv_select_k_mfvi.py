import argparse
import os
import time
from typing import List, Tuple

import numpy as np

from src.utils.mfvi_eval import infer_theta_single_row, compute_perplexity
from src.models.mfvi_model import MFVIModel
from src.utils.mfvi_monitor import ELBOMonitor


############################################
# Helper functions
############################################

def kfold_indices(n_samples: int, n_folds: int = 5, shuffle: bool = True, seed: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return train / validation indices for K-fold CV."""
    rng = np.random.default_rng(seed) if shuffle else None
    indices = np.arange(n_samples)
    if shuffle:
        rng.shuffle(indices)
    folds = []
    fold_sizes = [n_samples // n_folds] * n_folds
    for i in range(n_samples % n_folds):
        fold_sizes[i] += 1

    start = 0
    for size in fold_sizes:
        val_idx = indices[start : start + size]
        train_idx = np.setdiff1d(indices, val_idx, assume_unique=True)
        folds.append((train_idx, val_idx))
        start += size
    return folds


def train_mfvi(
    W: np.ndarray,
    alpha: np.ndarray,
    num_topics: int,
    max_iterations: int = 2000,
    convergence_threshold: float = 1e-6,
):
    """Train MFVI model and return trained instance."""
    model = MFVIModel(W, alpha, num_topics)
    monitor = ELBOMonitor(convergence_threshold, max_iterations)

    while True:
        elbo_before, elbo_after, param_changes = model.update_parameters()
        if monitor.check_convergence(elbo_before, elbo_after, param_changes):
            break
    return model


def cv_score_single_K(
    W: np.ndarray,
    alpha: np.ndarray,
    K: int,
    n_folds: int = 5,
    max_iterations: int = 2000,
    convergence_threshold: float = 1e-6,
) -> Tuple[float, List[float]]:
    """Perform K-fold CV for a single candidate K and return mean perplexity."""
    fold_scores = []
    folds = kfold_indices(W.shape[0], n_folds=n_folds, shuffle=True, seed=0)

    for train_idx, val_idx in folds:
        W_train = W[train_idx]
        W_val = W[val_idx]

        # Train model on training fold
        model = train_mfvi(
            W_train,
            alpha=np.ones(K + 1) / 10,
            num_topics=K + 1,  # +1 for healthy topic
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
        )
        beta_trained = model.beta  # (K+1, D)

        # Evaluate on validation fold
        P_pred = np.zeros_like(W_val, dtype=float)
        for i, row in enumerate(W_val):
            theta_exp, _ = infer_theta_single_row(
                W_row=row,
                beta=beta_trained,
                alpha=np.ones(K + 1) / 10,
                max_iterations=100,
                convergence_threshold=1e-4,
            )
            P_pred[i] = theta_exp @ beta_trained  # (D,)

        fold_perplexity = compute_perplexity(W_val, P_pred)
        fold_scores.append(fold_perplexity)

    mean_perp = float(np.mean(fold_scores))
    return mean_perp, fold_scores


############################################
# CLI
############################################

def parse_args():
    parser = argparse.ArgumentParser(description="Cross-validation based K selection for MFVI.")

    # Data generation / loading options
    parser.add_argument("--data_file", type=str, default=None, help="Path to .npy file containing binary matrix (subjects × diseases). If omitted synthetic data will be generated.")
    parser.add_argument("--M", type=int, default=500, help="Number of subjects for synthetic data")
    parser.add_argument("--D", type=int, default=100, help="Number of diseases for synthetic data")
    parser.add_argument("--true_K", type=int, default=10, help="Number of generative topics for synthetic data (excluding healthy)")
    parser.add_argument("--topic_prob", type=float, default=0.30)
    parser.add_argument("--nontopic_prob", type=float, default=0.01)

    # CV & algorithm parameters
    parser.add_argument("--candidate_Ks", type=str, default="5,10,15", help="Comma-separated list of candidate topic numbers (excluding healthy topic)")
    parser.add_argument("--n_folds", type=int, default=5)
    parser.add_argument("--max_iterations", type=int, default=2000)
    parser.add_argument("--convergence_threshold", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)

    # Output
    parser.add_argument("--output_csv", type=str, default=None, help="Optional path to write CSV summary of results")

    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    if args.data_file is not None:
        W = np.load(args.data_file)
        if not ((W == 0) | (W == 1)).all():
            raise ValueError("Data matrix must be binary (0/1).")
    else:
        from src.experiment.simulation import simulate_topic_disease_data

        W, _, _, _ = simulate_topic_disease_data(
            seed=args.seed,
            M=args.M,
            D=args.D,
            K=args.true_K,
            topic_associated_prob=args.topic_prob,
            nontopic_associated_prob=args.nontopic_prob,
            alpha=np.ones(args.true_K + 1) / 10,
            include_healthy_topic=True,
        )

    candidate_Ks = [int(k.strip()) for k in args.candidate_Ks.split(",") if k.strip()]

    results = []
    t0 = time.time()
    for K in candidate_Ks:
        mean_perp, fold_scores = cv_score_single_K(
            W,
            alpha=np.ones(K + 1) / 10,
            K=K,
            n_folds=args.n_folds,
            max_iterations=args.max_iterations,
            convergence_threshold=args.convergence_threshold,
        )
        results.append({
            "K": K,
            "mean_perplexity": mean_perp,
            "fold_perplexities": fold_scores,
        })
        print(f"K={K:3d} | mean perplexity={mean_perp:.4f}")

    elapsed = time.time() - t0
    print(f"\nCompleted CV in {elapsed:.1f}s\n")

    # Select best K
    best_entry = min(results, key=lambda x: x["mean_perplexity"])
    print(f"Best K according to mean perplexity: {best_entry['K']} (perplexity={best_entry['mean_perplexity']:.4f})")

    # Optionally write CSV
    if args.output_csv:
        import csv
        fieldnames = ["K", "mean_perplexity", "fold_perplexities"]
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        print(f"Results written to {args.output_csv}")


if __name__ == "__main__":
    main()
