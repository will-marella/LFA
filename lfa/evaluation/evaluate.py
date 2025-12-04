"""Core evaluation functions for comparing fitted results to ground truth.

This module wraps the existing validated metric computation logic from
lfa._experiment.get_metrics to provide a clean evaluation API.
"""

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Union, Optional

from lfa._core.results import LFAResult
from lfa._experiment.get_metrics import (
    align_mfvi_results,
    compute_mfvi_metrics,
    align_to_simulated_topics,
    compute_cgs_metrics
)


def evaluate_result(
    result: LFAResult,
    true_beta: np.ndarray,
    true_theta: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate fitted result against ground truth parameters.
    
    Aligns estimated topics to ground truth and computes recovery metrics.
    This function is for researchers conducting simulation studies where
    ground truth is known.
    
    Parameters
    ----------
    result : LFAResult
        Fitted model result from fit_lfa()
    true_beta : np.ndarray, shape (K+1, D)
        Ground truth topic-disease matrix (includes healthy topic)
    true_theta : np.ndarray, shape (M, K+1)
        Ground truth subject-topic distributions (includes healthy topic)
    
    Returns
    -------
    dict
        Metrics dictionary with algorithm-specific keys:
        
        MFVI metrics:
            - beta_correlation: Pearson correlation between aligned beta matrices
            - theta_correlation: Pearson correlation between aligned theta matrices
            - beta_mse: Mean squared error for beta
            - theta_mse: Mean squared error for theta
            - num_iterations: Number of iterations run
            - final_elbo: Final ELBO value
            - converged: Whether algorithm converged
            - mean_elbo_delta_tail: Mean ELBO change in tail window (if fixed_iterations=True)
        
        PCGS metrics:
            - beta_mae: Mean absolute error for beta
            - beta_pearson_corr: Pearson correlation for beta
            - theta_mae: Mean absolute error for theta
            - theta_pearson_corr: Pearson correlation for theta
            - num_iterations: Number of iterations run
            - r_hat_beta: Gelman-Rubin statistic for beta
            - r_hat_theta: Gelman-Rubin statistic for theta
            - r_hat_overall: Maximum R-hat across parameters
            - converged: Whether chains converged
    
    Examples
    --------
    >>> from lfa import fit_lfa, simulate_topic_disease_data
    >>> from lfa.evaluation import evaluate_result
    >>> 
    >>> # Simulate data with known ground truth
    >>> W, true_beta, true_theta, z = simulate_topic_disease_data(
    ...     seed=42, M=100, D=24, K=3
    ... )
    >>> 
    >>> # Fit model
    >>> result = fit_lfa(W, num_topics=3, algorithm='mfvi')
    >>> 
    >>> # Evaluate
    >>> metrics = evaluate_result(result, true_beta, true_theta)
    >>> print(f"Beta correlation: {metrics['beta_correlation']:.3f}")
    """
    # Dispatch to algorithm-specific evaluation
    if result.algorithm == 'mfvi':
        return _evaluate_mfvi(result, true_beta, true_theta)
    elif result.algorithm == 'pcgs':
        return _evaluate_pcgs(result, true_beta, true_theta)
    else:
        raise ValueError(f"Unknown algorithm: {result.algorithm}")


def _evaluate_mfvi(
    result: LFAResult,
    true_beta: np.ndarray,
    true_theta: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate MFVI result using existing metric logic.
    
    Aligns topics using Hungarian algorithm based on correlation,
    then computes MSE and correlation metrics.
    """
    # Convert LFAResult to legacy format expected by metric functions
    legacy_result = {
        'beta': result.beta.copy(),  # Copy to avoid modifying original
        'theta': result.theta.copy(),
        'z': result.z.copy(),
        'num_iterations': result.convergence_info.get('num_iterations'),
        'final_elbo': result.convergence_info.get('final_elbo')
    }
    
    # Align to ground truth (modifies legacy_result in-place)
    aligned = align_mfvi_results(legacy_result, true_beta)
    
    # Compute metrics using existing validated logic
    metrics = compute_mfvi_metrics(aligned, true_beta, true_theta)
    
    # Add convergence info from result
    metrics['converged'] = result.convergence_info.get('converged', False)
    if 'mean_elbo_delta_tail' in result.convergence_info:
        metrics['mean_elbo_delta_tail'] = result.convergence_info['mean_elbo_delta_tail']
    
    return metrics


def _evaluate_pcgs(
    result: LFAResult,
    true_beta: np.ndarray,
    true_theta: np.ndarray
) -> Dict[str, float]:
    """
    Evaluate PCGS result using existing metric logic.
    
    PCGS results are already aligned across chains (done in inference).
    This function aligns the merged result to ground truth and computes
    MAE and correlation metrics.
    """
    # Reconstruct combined_result format expected by alignment function
    combined_result = {
        'beta': result.beta.copy(),  # Copy to avoid modifying original
        'theta': result.theta.copy(),
        'z_distribution': result.z.copy()
    }
    
    # Align merged result to ground truth (modifies combined_result in-place)
    aligned = align_to_simulated_topics(combined_result, true_beta)
    
    # Compute metrics using existing validated logic
    metrics = compute_cgs_metrics(aligned, true_beta, true_theta)
    
    # Add convergence info from result
    metrics['num_iterations'] = result.convergence_info.get('num_iterations')
    metrics['r_hat_beta'] = result.convergence_info.get('r_hat_beta', float('inf'))
    metrics['r_hat_theta'] = result.convergence_info.get('r_hat_theta', float('inf'))
    metrics['r_hat_overall'] = result.convergence_info.get('r_hat_overall', float('inf'))
    metrics['converged'] = result.convergence_info.get('converged', False)
    
    return metrics


def compare_algorithms(
    W: Union[np.ndarray, pd.DataFrame],
    num_topics: int,
    true_beta: np.ndarray,
    true_theta: np.ndarray,
    algorithms: List[str] = ['mfvi', 'pcgs'],
    **algorithm_kwargs
) -> pd.DataFrame:
    """
    Fit multiple algorithms and compare their performance.
    
    Runs each specified algorithm on the same data and compares
    metrics against ground truth. Returns a DataFrame for easy analysis.
    
    Parameters
    ----------
    W : np.ndarray or pd.DataFrame, shape (M, D)
        Binary disease matrix
    num_topics : int
        Number of disease topics (excluding healthy topic)
    true_beta : np.ndarray, shape (K+1, D)
        Ground truth topic-disease matrix
    true_theta : np.ndarray, shape (M, K+1)
        Ground truth subject-topic distributions
    algorithms : list of str, default=['mfvi', 'pcgs']
        Algorithms to compare
    **algorithm_kwargs
        Algorithm-specific parameters. Prefix with algorithm name:
        - mfvi_max_iterations, mfvi_convergence_threshold, etc.
        - pcgs_num_chains, pcgs_max_iterations, etc.
    
    Returns
    -------
    pd.DataFrame
        Comparison results with columns:
        - algorithm: Algorithm name
        - runtime: Fitting time in seconds
        - converged: Whether algorithm converged
        - Algorithm-specific metric columns
    
    Examples
    --------
    >>> from lfa import simulate_topic_disease_data
    >>> from lfa.evaluation import compare_algorithms
    >>> 
    >>> # Simulate data
    >>> W, true_beta, true_theta, z = simulate_topic_disease_data(
    ...     seed=42, M=100, D=24, K=3
    ... )
    >>> 
    >>> # Compare algorithms
    >>> comparison = compare_algorithms(
    ...     W, num_topics=3,
    ...     true_beta=true_beta,
    ...     true_theta=true_theta,
    ...     algorithms=['mfvi', 'pcgs'],
    ...     mfvi_max_iterations=1000,
    ...     pcgs_num_chains=3,
    ...     pcgs_max_iterations=2000
    ... )
    >>> 
    >>> print(comparison[['algorithm', 'runtime', 'converged']])
    >>> comparison.to_csv('algorithm_comparison.csv', index=False)
    """
    # Import here to avoid circular dependency
    from lfa import fit_lfa
    
    results = []
    
    for algo in algorithms:
        # Extract algorithm-specific kwargs
        algo_kwargs = {}
        prefix = f"{algo}_"
        for key, value in algorithm_kwargs.items():
            if key.startswith(prefix):
                # Remove prefix (e.g., 'mfvi_max_iterations' -> 'max_iterations')
                algo_kwargs[key[len(prefix):]] = value
        
        # Fit model and time it
        start_time = time.time()
        result = fit_lfa(W, num_topics=num_topics, algorithm=algo, **algo_kwargs)
        runtime = time.time() - start_time
        
        # Evaluate against ground truth
        metrics = evaluate_result(result, true_beta, true_theta)
        
        # Build result row
        row = {
            'algorithm': algo,
            'runtime': runtime,
        }
        row.update(metrics)
        
        results.append(row)
    
    return pd.DataFrame(results)
