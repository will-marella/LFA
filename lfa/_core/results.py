"""LFAResult class for encapsulating LFA model fitting results."""

import numpy as np
from typing import Optional, Dict, Any, List, Union


class LFAResult:
    """
    Encapsulates results from fitting an LFA model.
    
    Contains fitted parameters (beta, theta, z) along with convergence 
    information and subject/disease identifiers.
    
    Attributes
    ----------
    beta : np.ndarray, shape (K+1, D)
        Topic-disease probability matrix. beta[k, d] is the probability 
        of disease d given topic k. Includes healthy topic.
        
    theta : np.ndarray, shape (M, K+1)
        Subject-topic weight matrix. theta[m, k] is the weight of topic k 
        for subject m. Rows sum to 1.
        
    z : np.ndarray
        Topic assignments for each subject-disease pair.
        MFVI: shape (M, D, K+1) with probabilistic assignments
        PCGS: shape (M, D, K+1) with merged distribution from chains
        
    num_topics : int
        Number of disease topics (K, excluding healthy topic)
        
    num_subjects : int
        Number of subjects (M)
        
    num_diseases : int
        Number of diseases (D)
        
    algorithm : str
        Algorithm used: 'mfvi' or 'pcgs'
        
    subject_ids : list
        Subject identifiers (length M)
        
    disease_names : list
        Disease names (length D)
        
    convergence_info : dict
        Algorithm-specific convergence information
        
    metadata : dict
        Hyperparameters and settings used for fitting
        
    chains : list of dict, optional
        For PCGS: aligned per-chain results (averaged over post-convergence window).
        Each dict contains 'beta', 'theta', 'z' for that chain.
        None for MFVI.
    """
    
    def __init__(
        self,
        beta: np.ndarray,
        theta: np.ndarray,
        z: np.ndarray,
        num_topics: int,
        algorithm: str,
        subject_ids: Optional[List] = None,
        disease_names: Optional[List] = None,
        convergence_info: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        chains: Optional[List[Dict[str, np.ndarray]]] = None
    ):
        """
        Initialize LFAResult.
        
        Parameters
        ----------
        beta : np.ndarray, shape (K+1, D)
            Topic-disease probability matrix
        theta : np.ndarray, shape (M, K+1)
            Subject-topic weight matrix
        z : np.ndarray
            Topic assignments
        num_topics : int
            Number of disease topics (excluding healthy)
        algorithm : str
            Algorithm used ('mfvi' or 'pcgs')
        subject_ids : list, optional
            Subject identifiers
        disease_names : list, optional
            Disease names
        convergence_info : dict, optional
            Convergence statistics
        metadata : dict, optional
            Hyperparameters and settings
        chains : list of dict, optional
            For PCGS: list of aligned per-chain results.
            Each dict has keys 'beta', 'theta', 'z'.
            None for MFVI.
        """
        # Validate inputs
        if beta.ndim != 2:
            raise ValueError(f"beta must be 2D, got shape {beta.shape}")
        if theta.ndim != 2:
            raise ValueError(f"theta must be 2D, got shape {theta.shape}")
        if z.ndim != 3:
            raise ValueError(f"z must be 3D, got shape {z.shape}")
            
        K_plus_1, D = beta.shape
        M, K_plus_1_theta = theta.shape
        M_z, D_z, K_plus_1_z = z.shape
        
        if K_plus_1 != K_plus_1_theta or K_plus_1 != K_plus_1_z:
            raise ValueError(f"Inconsistent K+1 dimensions: beta={K_plus_1}, theta={K_plus_1_theta}, z={K_plus_1_z}")
        if M != M_z:
            raise ValueError(f"Inconsistent M dimensions: theta={M}, z={M_z}")
        if D != D_z:
            raise ValueError(f"Inconsistent D dimensions: beta={D}, z={D_z}")
        if num_topics + 1 != K_plus_1:
            raise ValueError(f"num_topics={num_topics} but beta has {K_plus_1} topics (should be K+1)")
            
        if algorithm not in ['mfvi', 'pcgs']:
            raise ValueError(f"algorithm must be 'mfvi' or 'pcgs', got '{algorithm}'")
        
        # Store main results
        self.beta = beta
        self.theta = theta
        self.z = z
        self.num_topics = num_topics
        self.algorithm = algorithm
        
        # Store dimensions
        self.num_subjects = M
        self.num_diseases = D
        
        # Store identifiers (default to integer indices if not provided)
        self.subject_ids = subject_ids if subject_ids is not None else list(range(M))
        self.disease_names = disease_names if disease_names is not None else list(range(D))
        
        # Validate identifier lengths
        if len(self.subject_ids) != M:
            raise ValueError(f"subject_ids length ({len(self.subject_ids)}) must match M ({M})")
        if len(self.disease_names) != D:
            raise ValueError(f"disease_names length ({len(self.disease_names)}) must match D ({D})")
        
        # Store convergence info and metadata
        self.convergence_info = convergence_info if convergence_info is not None else {}
        self.metadata = metadata if metadata is not None else {}
        
        # Store per-chain results (PCGS only)
        self.chains = chains
        
        # Validate chains if provided
        if chains is not None:
            if algorithm != 'pcgs':
                raise ValueError("chains parameter only valid for PCGS algorithm")
            for i, chain in enumerate(chains):
                if not all(key in chain for key in ['beta', 'theta', 'z']):
                    raise ValueError(f"Chain {i} missing required keys (beta, theta, z)")
                # Check shapes match main arrays
                if chain['beta'].shape != beta.shape:
                    raise ValueError(f"Chain {i} beta shape {chain['beta'].shape} != merged beta shape {beta.shape}")
                if chain['theta'].shape != theta.shape:
                    raise ValueError(f"Chain {i} theta shape {chain['theta'].shape} != merged theta shape {theta.shape}")
                if chain['z'].shape != z.shape:
                    raise ValueError(f"Chain {i} z shape {chain['z'].shape} != merged z shape {z.shape}")
    
    def summary(self) -> str:
        """
        Generate a human-readable summary of the fitted model.
        
        Returns
        -------
        str
            Summary text
        """
        lines = []
        lines.append("=" * 60)
        lines.append("LFA Model Results Summary")
        lines.append("=" * 60)
        lines.append("")
        
        # Basic info
        lines.append(f"Algorithm: {self.algorithm.upper()}")
        lines.append(f"Number of subjects: {self.num_subjects}")
        lines.append(f"Number of diseases: {self.num_diseases}")
        lines.append(f"Number of disease topics: {self.num_topics} (+ 1 healthy topic)")
        lines.append("")
        
        # Convergence info
        lines.append("Convergence Information:")
        if self.algorithm == 'mfvi':
            lines.append(f"  Iterations: {self.convergence_info.get('num_iterations', 'N/A')}")
            lines.append(f"  Final ELBO: {self.convergence_info.get('final_elbo', 'N/A'):.2f}")
            final_change = self.convergence_info.get('final_elbo_change', 'N/A')
            if isinstance(final_change, float):
                lines.append(f"  Final ELBO change: {final_change:.6f}")
            lines.append(f"  Converged: {self.convergence_info.get('converged', 'N/A')}")
        elif self.algorithm == 'pcgs':
            lines.append(f"  Iterations: {self.convergence_info.get('num_iterations', 'N/A')}")
            num_chains = len(self.chains) if self.chains is not None else self.convergence_info.get('num_chains', 'N/A')
            lines.append(f"  Number of chains: {num_chains}")
            lines.append(f"  R-hat (beta): {self.convergence_info.get('r_hat_beta', 'N/A'):.4f}")
            lines.append(f"  R-hat (theta): {self.convergence_info.get('r_hat_theta', 'N/A'):.4f}")
            lines.append(f"  Converged: {self.convergence_info.get('converged', 'N/A')}")
            if self.chains is not None:
                lines.append(f"  Per-chain results available: Yes ({len(self.chains)} chains)")
        lines.append("")
        
        # Topic prevalence
        lines.append("Topic Prevalence (mean theta across subjects):")
        topic_prevalence = self.theta.mean(axis=0)
        for k in range(len(topic_prevalence)):
            topic_label = f"Topic {k}" if k < self.num_topics else "Healthy Topic"
            lines.append(f"  {topic_label}: {topic_prevalence[k]:.3f}")
        lines.append("")
        
        # Disease coverage
        lines.append("Disease Statistics:")
        disease_in_topic = (self.beta > 0.5).sum(axis=0)
        avg_diseases_per_topic = disease_in_topic.mean()
        lines.append(f"  Avg diseases per topic (>0.5 prob): {avg_diseases_per_topic:.1f}")
        lines.append("")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"LFAResult(algorithm='{self.algorithm}', "
                f"num_subjects={self.num_subjects}, "
                f"num_diseases={self.num_diseases}, "
                f"num_topics={self.num_topics})")
    
    def get_top_diseases_per_topic(
        self, 
        n: int = 10, 
        use_names: bool = True,
        exclude_healthy: bool = True
    ) -> Dict[int, List[tuple]]:
        """
        Get the top N diseases most associated with each topic.
        
        This is the primary method for interpreting what each disease topic
        represents. Diseases are ranked by their probability beta[k, d] for
        each topic k.
        
        Parameters
        ----------
        n : int, default=10
            Number of top diseases to return per topic
        use_names : bool, default=True
            If True, return disease names; if False, return disease indices
        exclude_healthy : bool, default=True
            If True, exclude the healthy topic (topic 0) from results
        
        Returns
        -------
        dict
            Dictionary mapping topic index to list of (disease, probability) tuples.
            Topics are sorted by index. Within each topic, diseases are sorted
            by probability (highest first).
        
        Examples
        --------
        >>> result = fit_lfa(W, num_topics=3, algorithm='mfvi')
        >>> top_diseases = result.get_top_diseases_per_topic(n=5)
        >>> for topic_idx, diseases in top_diseases.items():
        ...     print(f"Topic {topic_idx}:")
        ...     for disease, prob in diseases:
        ...         print(f"  {disease}: {prob:.3f}")
        Topic 1:
          Diabetes: 0.856
          Hypertension: 0.723
          Obesity: 0.691
          ...
        """
        results = {}
        
        # Determine topic range
        start_topic = 1 if exclude_healthy else 0
        
        for k in range(start_topic, self.beta.shape[0]):
            # Get probabilities for this topic
            topic_probs = self.beta[k, :]
            
            # Get top N disease indices (sorted descending)
            top_indices = np.argsort(topic_probs)[-n:][::-1]
            
            # Build list of (disease, probability) tuples
            if use_names:
                top_items = [
                    (self.disease_names[i], topic_probs[i]) 
                    for i in top_indices
                ]
            else:
                top_items = [
                    (i, topic_probs[i]) 
                    for i in top_indices
                ]
            
            # Use 1-indexed topic labels for user-facing output (skip healthy topic)
            topic_label = k if not exclude_healthy else k
            results[topic_label] = top_items
        
        return results
    
    def get_disease_topic_loadings(
        self,
        disease_name: Optional[Union[str, int]] = None,
        disease_idx: Optional[int] = None
    ) -> np.ndarray:
        """
        Get topic loadings (beta) for a specific disease.
        
        Shows which topics are most strongly associated with this disease.
        Useful for understanding the latent structure explaining a particular
        condition.
        
        Parameters
        ----------
        disease_name : str or int, optional
            Name or ID of disease (must exist in disease_names)
        disease_idx : int, optional
            Index of disease (0-based)
        
        Returns
        -------
        np.ndarray, shape (K+1,)
            Topic probabilities for this disease. Entry [k] is beta[k, disease].
            Includes healthy topic at index 0.
        
        Raises
        ------
        ValueError
            If neither or both parameters provided, or if disease not found
        
        Examples
        --------
        >>> result = fit_lfa(W, num_topics=3, algorithm='mfvi')
        >>> loadings = result.get_disease_topic_loadings(disease_name='Diabetes')
        >>> print(f"Diabetes is most associated with topic {loadings.argmax()}")
        Diabetes is most associated with topic 2
        
        >>> # Get probabilities excluding healthy topic
        >>> disease_loadings = loadings[1:]  # Skip index 0 (healthy topic)
        >>> print(f"Strongest disease topic: {disease_loadings.argmax() + 1}")
        Strongest disease topic: 2
        """
        # Validate inputs
        if disease_name is None and disease_idx is None:
            raise ValueError("Must provide either disease_name or disease_idx")
        if disease_name is not None and disease_idx is not None:
            raise ValueError("Cannot provide both disease_name and disease_idx")
        
        # Resolve to index
        if disease_name is not None:
            try:
                idx = self.disease_names.index(disease_name)  # type: ignore
            except ValueError:
                raise ValueError(f"Disease '{disease_name}' not found in disease_names. "
                               f"Available diseases: {self.disease_names}")
        else:
            # disease_idx provided
            if disease_idx is not None and (disease_idx < 0 or disease_idx >= self.num_diseases):
                raise ValueError(f"disease_idx {disease_idx} out of range [0, {self.num_diseases})")
            idx = disease_idx
        
        # Return beta[:, idx]
        return self.beta[:, idx].copy()
    
    def to_dict(self, include_chains=True) -> Dict[str, Any]:
        """
        Convert result to dictionary for serialization.
        
        Parameters
        ----------
        include_chains : bool, default=True
            If True, include per-chain results for PCGS (can be large)
            
        Returns
        -------
        dict
            Dictionary containing all result data
        """
        result_dict = {
            'beta': self.beta,
            'theta': self.theta,
            'z': self.z,
            'num_topics': self.num_topics,
            'num_subjects': self.num_subjects,
            'num_diseases': self.num_diseases,
            'algorithm': self.algorithm,
            'subject_ids': self.subject_ids,
            'disease_names': self.disease_names,
            'convergence_info': self.convergence_info,
            'metadata': self.metadata,
        }
        
        if include_chains and self.chains is not None:
            result_dict['chains'] = self.chains
            
        return result_dict
