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
