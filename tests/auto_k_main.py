import pathlib, sys, os

# Ensure that the project root/LFA directory is on PYTHONPATH even when the
# script is launched from the repository root.
_LFA_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_LFA_ROOT) not in sys.path:
    sys.path.insert(0, str(_LFA_ROOT))

import numpy as np

from src.experiment.simulation import simulate_topic_disease_data
from src.utils.auto_k_mfvi import select_k_mfvi

if __name__ == "__main__":
    # Simulation parameters (keep modest for quick test)
    M = 500  # number of subjects
    D = 24    # number of diseases
    true_K = 8
    seed = 0
    alpha_sim = np.ones(true_K + 1) / 10

    # Generate synthetic dataset
    W, z, beta, theta = simulate_topic_disease_data(
        seed=seed,
        M=M,
        D=D,
        K=true_K,
        topic_associated_prob=0.30,
        nontopic_associated_prob=0.01,
        alpha=alpha_sim,
        include_healthy_topic=True,
    )

    # Candidate Ks to test (excluding healthy)
    candidate_Ks = [4, 8, 12, 16, 20]

    # Run automatic K selection
    best_k, results = select_k_mfvi(
        W,
        candidate_Ks=candidate_Ks,
        n_folds=4,
        max_iterations=3000,
        convergence_threshold=0.01,
    )

    print("Results per K (mean perplexity):")
    for entry in results:
        print(f"  K={entry['K']:2d} -> {entry['mean_perplexity']:.4f}")

    print("\nChosen best K:", best_k)
