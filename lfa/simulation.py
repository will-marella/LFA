"""Data simulation for LFA models.

This module provides functions for generating synthetic disease data
for testing and validation.
"""

import numpy as np
from lfa._experiment.simulation import simulate_topic_disease_data as _sim

def simulate_topic_disease_data(
    seed: int,
    M: int,
    D: int,
    K: int,
    topic_associated_prob: float = 0.30,
    nontopic_associated_prob: float = 0.01,
    alpha: np.ndarray = None,
    include_healthy_topic: bool = True
) -> tuple:
    """
    Simulate disease data for LFA model testing.
    
    Parameters
    ----------
    seed : int
        Random seed for reproducibility
    M : int
        Number of subjects
    D : int
        Number of diseases
    K : int
        Number of disease topics (excluding healthy topic)
    topic_associated_prob : float, default=0.30
        Probability of disease for topic-associated diseases
    nontopic_associated_prob : float, default=0.01
        Probability of disease for non-associated diseases
    alpha : np.ndarray, optional
        Dirichlet prior (K+1,). If None, defaults to 0.1 uniform
    include_healthy_topic : bool, default=True
        Whether to include healthy topic (always True for LFA)
    
    Returns
    -------
    tuple: (W, beta, theta, z)
        W : np.ndarray, shape (M, D)
            Binary disease matrix
        beta : np.ndarray, shape (K+1, D)
            True topic-disease probabilities (includes healthy topic)
        theta : np.ndarray, shape (M, K+1)
            True subject-topic distributions (includes healthy topic)
        z : np.ndarray, shape (M, D)
            True topic assignments for each subject-disease pair
    
    Examples
    --------
    >>> from lfa import simulate_topic_disease_data, fit_lfa
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
    >>> # Evaluate against ground truth
    >>> metrics = evaluate_result(result, true_beta, true_theta)
    """
    # Call internal simulation (returns W, z, beta, theta)
    W, z, beta, theta = _sim(
        seed=seed,
        M=M,
        D=D,
        K=K,
        topic_associated_prob=topic_associated_prob,
        nontopic_associated_prob=nontopic_associated_prob,
        alpha=alpha,
        include_healthy_topic=include_healthy_topic
    )
    
    # Return in more intuitive order: W, beta, theta, z
    return W, beta, theta, z

__all__ = ['simulate_topic_disease_data']
