"""Main user-facing API for Latent Factor Allocation (LFA).

This module provides clean, user-friendly functions for fitting LFA models
to binary disease data.
"""

import numpy as np
import warnings
from typing import Union, Optional, List, Tuple, Dict, Any

from lfa._core.inference import run_mfvi_inference, run_pcgs_inference
from lfa._core.results import LFAResult
from lfa._utils.auto_k_mfvi import select_k_mfvi


def fit_lfa(
    W: Union[np.ndarray, 'pd.DataFrame'],
    num_topics: int,
    algorithm: str = 'mfvi',
    alpha: Optional[Union[float, np.ndarray]] = None,
    subject_ids: Optional[List] = None,
    disease_names: Optional[List] = None,
    max_iterations: Optional[int] = None,
    convergence_threshold: Optional[float] = None,
    num_chains: int = 3,
    verbose: bool = True,
    **algorithm_kwargs
) -> LFAResult:
    """
    Fit a Latent Factor Allocation model to binary disease data.
    
    This is the main entry point for users. Fits an LFA model using either
    MFVI (fast, approximate) or PCGS (slower, exact) inference.
    
    Parameters
    ----------
    W : np.ndarray or pd.DataFrame, shape (M, D)
        Binary disease matrix where M = subjects, D = diseases.
        Values must be 0 (disease absent) or 1 (disease present).
        If DataFrame, row/column labels are automatically extracted.
        
    num_topics : int
        Number of disease topics to infer (excluding healthy topic).
        The model will fit K disease topics + 1 healthy topic.
        
    algorithm : {'mfvi', 'pcgs'}, default='mfvi'
        Inference algorithm:
        - 'mfvi': Mean Field Variational Inference (fast, scalable)
        - 'pcgs': Partially Collapsed Gibbs Sampling (slow, exact)
        
    alpha : float or np.ndarray, optional
        Dirichlet prior concentration parameter.
        - If float: uniform prior with value alpha for all topics
        - If array: must have shape (num_topics+1,)
        - Default: 0.1 (weakly informative uniform prior)
        
    subject_ids : list, optional
        Identifiers for subjects (length M).
        If W is DataFrame, automatically extracted from index.
        If None, uses integer indices.
        
    disease_names : list, optional
        Names for diseases (length D).
        If W is DataFrame, automatically extracted from columns.
        If None, uses integer indices.
        
    max_iterations : int, optional
        Maximum number of iterations.
        Default: 1000 for MFVI, 3000 for PCGS.
        
    convergence_threshold : float, optional
        Convergence threshold.
        Default: 1e-6 for MFVI (ELBO change), unused for PCGS (uses R-hat).
        
    num_chains : int, default=3
        Number of chains for PCGS (ignored for MFVI).
        
    verbose : bool, default=True
        Whether to print progress information.
        
    **algorithm_kwargs : dict
        Additional algorithm-specific parameters:
        MFVI: fixed_iterations, delta_tail_window
        PCGS: window_size, r_hat_threshold, post_convergence_samples, base_seed
        
    Returns
    -------
    LFAResult
        Object containing fitted parameters (beta, theta, z) and analysis methods.
        
    Raises
    ------
    ValueError
        If W is not binary, has wrong shape, or parameters are invalid.
    RuntimeError
        If fitting fails.
        
    Examples
    --------
    Basic usage with NumPy array:
    
    >>> import numpy as np
    >>> from lfa import fit_lfa
    >>> W = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 1]])
    >>> result = fit_lfa(W, num_topics=2, algorithm='mfvi')
    >>> print(result.summary())
    
    With pandas DataFrame and disease names:
    
    >>> import pandas as pd
    >>> W_df = pd.DataFrame(
    ...     [[0, 1, 0], [1, 0, 1]],
    ...     columns=['Diabetes', 'Hypertension', 'Asthma']
    ... )
    >>> result = fit_lfa(W_df, num_topics=2)
    >>> result.beta  # Topic-disease probabilities with disease names
    
    Using PCGS for small dataset:
    
    >>> result = fit_lfa(W, num_topics=2, algorithm='pcgs', num_chains=4)
    >>> result.chains  # Access per-chain results
    """
    # Extract DataFrame info if provided
    try:
        import pandas as pd
        is_dataframe = isinstance(W, pd.DataFrame)
    except ImportError:
        is_dataframe = False
    
    if is_dataframe:
        # Extract labels from DataFrame
        if subject_ids is None:
            subject_ids = W.index.tolist()
        if disease_names is None:
            disease_names = W.columns.tolist()
        # Convert to numpy array
        W_array = W.values
    else:
        W_array = np.asarray(W)
    
    # Validate input
    if W_array.ndim != 2:
        raise ValueError(f"W must be 2D, got shape {W_array.shape}")
    
    M, D = W_array.shape
    
    # Check binary
    if not np.all(np.isin(W_array, [0, 1])):
        raise ValueError("W must be binary (only 0 and 1 values)")
    
    # Check for NaN
    if np.isnan(W_array).any():
        raise ValueError("W contains missing values (NaN). Please impute before fitting.")
    
    # Validate num_topics
    if num_topics < 1:
        raise ValueError(f"num_topics must be >= 1, got {num_topics}")
    if num_topics >= D:
        warnings.warn(f"num_topics ({num_topics}) >= num_diseases ({D}). "
                     f"Consider using fewer topics for better interpretability.")
    
    # Prepare alpha
    K_total = num_topics + 1  # Include healthy topic
    if alpha is None:
        alpha = np.ones(K_total) * 0.1
    elif isinstance(alpha, (int, float)):
        alpha = np.ones(K_total) * alpha
    else:
        alpha = np.asarray(alpha)
        if alpha.shape != (K_total,):
            raise ValueError(f"alpha must have shape ({K_total},), got {alpha.shape}")
    
    # Prepare subject_ids and disease_names
    if subject_ids is None:
        subject_ids = list(range(M))
    elif len(subject_ids) != M:
        raise ValueError(f"subject_ids length ({len(subject_ids)}) must match M ({M})")
    
    if disease_names is None:
        disease_names = list(range(D))
    elif len(disease_names) != D:
        raise ValueError(f"disease_names length ({len(disease_names)}) must match D ({D})")
    
    # Set default parameters based on algorithm
    if algorithm == 'mfvi':
        if max_iterations is None:
            max_iterations = 1000
        if convergence_threshold is None:
            convergence_threshold = 1e-6
            
        # Run MFVI
        result = run_mfvi_inference(
            W=W_array,
            alpha=alpha,
            num_topics=K_total,
            max_iterations=max_iterations,
            convergence_threshold=convergence_threshold,
            verbose=verbose,
            **algorithm_kwargs
        )
        
    elif algorithm == 'pcgs':
        if max_iterations is None:
            max_iterations = 3000
            
        # Run PCGS
        result = run_pcgs_inference(
            W=W_array,
            alpha=alpha,
            num_topics=K_total,
            num_chains=num_chains,
            max_iterations=max_iterations,
            verbose=verbose,
            **algorithm_kwargs
        )
        
    else:
        raise ValueError(f"algorithm must be 'mfvi' or 'pcgs', got '{algorithm}'")
    
    # Add subject and disease labels to result
    result.subject_ids = subject_ids
    result.disease_names = disease_names
    
    return result


def select_num_topics(
    W: Union[np.ndarray, 'pd.DataFrame'],
    candidate_topics: Optional[List[int]] = None,
    algorithm: str = 'mfvi',
    alpha: Optional[Union[float, np.ndarray]] = None,
    n_folds: int = 5,
    metric: str = 'bic',
    subject_ids: Optional[List] = None,
    disease_names: Optional[List] = None,
    verbose: bool = True,
    **algorithm_kwargs
) -> Tuple[int, List[Dict[str, Any]]]:
    """
    Automatically select the optimal number of topics using cross-validation.
    
    Note: Current BIC implementation needs validation. Use results with caution
    and consider trying multiple K values manually.
    
    Parameters
    ----------
    W : np.ndarray or pd.DataFrame, shape (M, D)
        Binary disease matrix.
        
    candidate_topics : list of int, optional
        List of K values to evaluate.
        Default: [3, 5, 8, 10, 15, 20]
        
    algorithm : str, default='mfvi'
        Only 'mfvi' is currently supported (PCGS too slow for CV).
        
    alpha : float or np.ndarray, optional
        Dirichlet prior (default: 0.1).
        
    n_folds : int, default=5
        Number of cross-validation folds.
        
    metric : {'bic', 'perplexity'}, default='bic'
        Metric for selection. Note: BIC implementation needs validation.
        
    subject_ids : list, optional
        Subject identifiers.
        
    disease_names : list, optional
        Disease names.
        
    verbose : bool, default=True
        Print progress.
        
    **algorithm_kwargs : dict
        Additional parameters for fitting.
        
    Returns
    -------
    best_k : int
        Optimal number of topics (excluding healthy topic).
        
    cv_results : list of dict
        Cross-validation results for each candidate K.
        Each dict contains: 'K', 'mean_perplexity', 'mean_bic', 
        'fold_perplexities', 'fold_bics'.
        
    Raises
    ------
    ValueError
        If algorithm is not 'mfvi' or inputs are invalid.
        
    Examples
    --------
    >>> from lfa import select_num_topics, fit_lfa
    >>> best_k, results = select_num_topics(W, candidate_topics=[3, 5, 8, 10])
    >>> print(f"Best K: {best_k}")
    >>> result = fit_lfa(W, num_topics=best_k)
    """
    # Only MFVI supported for auto-K
    if algorithm != 'mfvi':
        raise ValueError("Only 'mfvi' algorithm is currently supported for auto-K selection")
    
    # Extract DataFrame info if provided
    try:
        import pandas as pd
        is_dataframe = isinstance(W, pd.DataFrame)
    except ImportError:
        is_dataframe = False
    
    if is_dataframe:
        W_array = W.values
    else:
        W_array = np.asarray(W)
    
    # Validate input
    if W_array.ndim != 2:
        raise ValueError(f"W must be 2D, got shape {W_array.shape}")
    
    if not np.all(np.isin(W_array, [0, 1])):
        raise ValueError("W must be binary (only 0 and 1 values)")
    
    # Default candidate topics
    if candidate_topics is None:
        candidate_topics = [3, 5, 8, 10, 15, 20]
    
    # Warn about BIC
    if metric == 'bic' and verbose:
        warnings.warn(
            "BIC-based selection is experimental and needs validation. "
            "Consider manually trying different K values.",
            UserWarning
        )
    
    # Prepare alpha (will be adjusted per K in select_k_mfvi)
    alpha_value = 0.1 if alpha is None else alpha
    
    # Extract algorithm-specific parameters
    max_iterations = algorithm_kwargs.pop('max_iterations', 2000)
    convergence_threshold = algorithm_kwargs.pop('convergence_threshold', 1e-6)
    inference_max_iter = algorithm_kwargs.pop('inference_max_iter', 100)
    inference_tol = algorithm_kwargs.pop('inference_tol', 1e-4)
    seed = algorithm_kwargs.pop('seed', 0)
    
    # Run CV-based K selection
    best_k, cv_results = select_k_mfvi(
        W=W_array,
        candidate_Ks=candidate_topics,
        alpha=None if isinstance(alpha_value, (int, float)) else alpha_value,
        n_folds=n_folds,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        inference_max_iter=inference_max_iter,
        inference_tol=inference_tol,
        seed=seed
    )
    
    if verbose:
        print("\nCross-validation results:")
        print(f"{'K':<5} {'Mean BIC':<12} {'Mean Perplexity':<18}")
        print("-" * 40)
        for result in cv_results:
            print(f"{result['K']:<5} {result['mean_bic']:<12.2f} {result['mean_perplexity']:<18.4f}")
        print(f"\nBest K (minimum BIC): {best_k}")
    
    return best_k, cv_results
