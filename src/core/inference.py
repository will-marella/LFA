"""Clean inference wrappers for MFVI and PCGS algorithms.

These functions wrap the existing validated implementations without requiring
ground truth parameters, making them suitable for end users.
"""

import time
import numpy as np
from typing import Optional, Dict, Any, List

from src.models.mfvi_model import MFVIModel
from src.models.pcgs import collapsed_gibbs_sampling
from src.utils.mfvi_monitor import ELBOMonitor
from src.experiment.get_metrics import align_chains, merge_chains
from src.core.results import LFAResult


def run_mfvi_inference(
    W: np.ndarray,
    alpha: np.ndarray,
    num_topics: int,
    max_iterations: int = 1000,
    convergence_threshold: float = 1e-6,
    fixed_iterations: bool = False,
    delta_tail_window: int = 50,
    verbose: bool = True
) -> LFAResult:
    """
    Run MFVI inference on binary disease data.
    
    Fits an LFA model using Mean Field Variational Inference. Returns
    fitted parameters (beta, theta, z) with convergence diagnostics.
    
    Parameters
    ----------
    W : np.ndarray, shape (M, D)
        Binary disease matrix
    alpha : np.ndarray, shape (K+1,)
        Dirichlet prior concentration parameter (includes healthy topic)
    num_topics : int
        Total number of topics (K+1, including healthy topic)
    max_iterations : int, default=1000
        Maximum number of iterations
    convergence_threshold : float, default=1e-6
        ELBO convergence threshold
    fixed_iterations : bool, default=False
        If True, always run max_iterations
    delta_tail_window : int, default=50
        Window size for averaging ELBO changes
    verbose : bool, default=True
        Print progress information
        
    Returns
    -------
    LFAResult
        Fitted model results
    """
    start_time = time.time()
    
    # Initialize model and monitor
    model = MFVIModel(W, alpha, num_topics)
    monitor = ELBOMonitor(
        convergence_threshold=convergence_threshold,
        max_iterations=max_iterations,
        force_max_iterations=fixed_iterations,
        delta_tail_window=delta_tail_window
    )
    
    # Run iterations until convergence
    while True:
        elbo_before, elbo_after, param_changes = model.update_parameters()
        
        if monitor.check_convergence(elbo_before, elbo_after, param_changes):
            break
    
    end_time = time.time()
    run_time = end_time - start_time
    
    # Get convergence statistics
    convergence_stats = monitor.get_convergence_stats()
    
    # Normalize theta for final result
    theta_normalized = model.e_theta / np.sum(model.e_theta, axis=1, keepdims=True)
    
    # Calculate final ELBO change
    elbo_history = convergence_stats['elbo_history']
    final_elbo_change = abs(elbo_history[-1] - elbo_history[-2]) if len(elbo_history) >= 2 else 0.0
    
    # Prepare convergence info
    convergence_info = {
        'num_iterations': convergence_stats['num_iterations'],
        'final_elbo': convergence_stats['final_elbo'],
        'final_elbo_change': final_elbo_change,
        'converged': convergence_stats['converged'],
        'elbo_history': convergence_stats['elbo_history'],
        'mean_elbo_delta_tail': convergence_stats.get('mean_elbo_delta_tail'),
        'run_time': run_time
    }
    
    # Prepare metadata
    metadata = {
        'alpha': alpha.tolist(),
        'max_iterations': max_iterations,
        'convergence_threshold': convergence_threshold,
        'fixed_iterations': fixed_iterations,
        'delta_tail_window': delta_tail_window
    }
    
    # Create LFAResult
    result = LFAResult(
        beta=model.beta,
        theta=theta_normalized,
        z=model.z,
        num_topics=num_topics - 1,  # Exclude healthy topic from user-facing count
        algorithm='mfvi',
        convergence_info=convergence_info,
        metadata=metadata,
        chains=None  # MFVI doesn't have chains
    )
    
    if verbose:
        print(f"\nMFVI completed in {run_time:.2f} seconds")
        print(f"Iterations: {convergence_stats['num_iterations']}")
        print(f"Final ELBO: {convergence_stats['final_elbo']:.2f}")
        print(f"Converged: {convergence_stats['converged']}")
    
    return result


def run_pcgs_inference(
    W: np.ndarray,
    alpha: np.ndarray,
    num_topics: int,
    num_chains: int = 3,
    max_iterations: int = 3000,
    window_size: int = 500,
    r_hat_threshold: float = 1.1,
    post_convergence_samples: int = 50,
    base_seed: Optional[int] = None,
    verbose: bool = True
) -> LFAResult:
    """
    Run PCGS inference on binary disease data.
    
    Fits an LFA model using Partially Collapsed Gibbs Sampling. Returns
    fitted parameters (beta, theta, z) from multiple chains with R-hat diagnostics.
    
    Parameters
    ----------
    W : np.ndarray, shape (M, D)
        Binary disease matrix
    alpha : np.ndarray, shape (K+1,)
        Dirichlet prior concentration parameter (includes healthy topic)
    num_topics : int
        Total number of topics (K+1, including healthy topic)
    num_chains : int, default=3
        Number of parallel Markov chains
    max_iterations : int, default=3000
        Maximum iterations per chain
    window_size : int, default=500
        Window for R-hat calculation
    r_hat_threshold : float, default=1.1
        Convergence threshold for R-hat
    post_convergence_samples : int, default=50
        Samples to collect after convergence
    base_seed : int, optional
        Random seed for reproducibility
    verbose : bool, default=True
        Print progress information
        
    Returns
    -------
    LFAResult
        Fitted model results with per-chain and merged parameters
    """
    start_time = time.time()
    
    # Run PCGS
    results = collapsed_gibbs_sampling(
        W=W,
        alpha=alpha,
        num_topics=num_topics,
        num_chains=num_chains,
        max_iterations=max_iterations,
        window_size=window_size,
        r_hat_threshold=r_hat_threshold,
        calculate_ess=False,
        ess_threshold=400,
        post_convergence_samples=post_convergence_samples,
        base_seed=base_seed
    )
    
    end_time = time.time()
    run_time = end_time - start_time
    
    # Extract results
    chain_results = results.get('chain_results', [])
    monitor_stats = results.get('monitor_stats', {})
    
    if not chain_results:
        raise RuntimeError("PCGS failed: no chain results returned")
    
    # Align chains to each other
    aligned_results = align_chains(chain_results)
    
    # Merge chains for main result
    merged_result = merge_chains(aligned_results)
    
    # Extract per-chain averages (aligned)
    chains_list = []
    for chain in aligned_results:
        # Average samples from each chain
        chain_beta = np.mean(np.array(chain['beta_samples']), axis=0)
        chain_theta = np.mean(np.array(chain['theta_samples']), axis=0)
        
        # For z, compute distribution over samples
        z_samples = np.array(chain['z_samples'])
        M, D = z_samples[0].shape
        z_distribution = np.zeros((M, D, num_topics))
        for sample in z_samples:
            for m in range(M):
                for d in range(D):
                    z_distribution[m, d, sample[m, d]] += 1
        z_distribution = z_distribution / len(z_samples)
        
        chains_list.append({
            'beta': chain_beta,
            'theta': chain_theta,
            'z': z_distribution
        })
    
    # Prepare convergence info
    r_hat_beta = monitor_stats.get('r_hats', {}).get('beta', float('inf'))
    r_hat_theta = monitor_stats.get('r_hats', {}).get('theta', float('inf'))
    r_hat_overall = max(r_hat_beta, r_hat_theta)
    converged = r_hat_overall < r_hat_threshold
    
    num_iterations_per_chain = monitor_stats.get('chain_iterations', {})
    max_iters = max(num_iterations_per_chain.values()) if num_iterations_per_chain else max_iterations
    
    convergence_info = {
        'num_iterations': max_iters,
        'num_chains': num_chains,
        'r_hat_beta': r_hat_beta,
        'r_hat_theta': r_hat_theta,
        'r_hat_overall': r_hat_overall,
        'converged': converged,
        'num_samples_collected': post_convergence_samples,
        'run_time': run_time
    }
    
    # Prepare metadata
    metadata = {
        'alpha': alpha.tolist(),
        'num_chains': num_chains,
        'max_iterations': max_iterations,
        'window_size': window_size,
        'r_hat_threshold': r_hat_threshold,
        'post_convergence_samples': post_convergence_samples,
        'base_seed': base_seed
    }
    
    # Create LFAResult
    result = LFAResult(
        beta=merged_result['beta'],
        theta=merged_result['theta'],
        z=merged_result['z'],
        num_topics=num_topics - 1,  # Exclude healthy topic from user-facing count
        algorithm='pcgs',
        convergence_info=convergence_info,
        metadata=metadata,
        chains=chains_list
    )
    
    if verbose:
        print(f"\nPCGS completed in {run_time:.2f} seconds")
        print(f"Iterations: {max_iters}")
        print(f"R-hat (beta): {r_hat_beta:.4f}, R-hat (theta): {r_hat_theta:.4f}")
        print(f"Converged: {converged}")
        print(f"Chains available: {len(chains_list)}")
    
    return result
