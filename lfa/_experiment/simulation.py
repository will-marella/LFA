import numpy as np


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
    Simulate disease occurrence data based on a topic model approach.
    
    Parameters:
    -----------
    seed : int
        Random seed for reproducibility
    M : int
        Number of subjects
    D : int
        Number of diseases
    K : int
        Number of topics (excluding healthy topic if include_healthy_topic=True)
    topic_associated_prob : float
        Probability of disease occurrence for diseases associated with a topic
    nontopic_associated_prob : float
        Probability of disease occurrence for diseases not associated with a topic
    alpha : np.ndarray
        Dirichlet parameter for topic distributions. If None, defaults to 1/10 for all topics
    include_healthy_topic : bool
        Whether to include a "healthy" topic with nontopic_associated_prob for all diseases
        
    Returns:
    --------
    tuple: (W, z, beta, theta)
        W : np.ndarray (M x D)
            Binary matrix of disease occurrences
        z : np.ndarray (M x D)
            Topic assignments for each subject-disease pair
        beta : np.ndarray (K x D)
            Disease probability matrix for each topic
        theta : np.ndarray (M x K)
            Topic distribution for each subject
    """
    # Set random seed
    np.random.seed(seed)
    
    # Adjust K if including healthy topic
    total_K = K + 1 if include_healthy_topic else K
    
    # Check if D is evenly divisible by K
    if D % K != 0:
        raise ValueError(f"Number of diseases (D={D}) must be evenly divisible by number of topics (K={K})")
    
    # Set up alpha if not provided
    if alpha is None:
        alpha = np.ones(total_K) / 10
    elif len(alpha) != total_K:
        raise ValueError(f"Alpha must have length {total_K} (got {len(alpha)})")
    
    # Create beta matrix
    beta = np.zeros((total_K, D))
    
    # If including healthy topic, set first row
    current_pos = 0
    if include_healthy_topic:
        beta[0,:] = nontopic_associated_prob
        start_k = 1
    else:
        start_k = 0
    
    # Distribute diseases among topics
    diseases_per_topic = D // K
    for k in range(start_k, total_K):
        topic_diseases = np.zeros(D)
        topic_diseases[current_pos:current_pos + diseases_per_topic] = topic_associated_prob
        topic_diseases[topic_diseases == 0] = nontopic_associated_prob
        beta[k,:] = topic_diseases
        current_pos += diseases_per_topic
    
    # Simulate topic distributions
    theta = np.random.dirichlet(alpha, M)
    
    # Simulate disease occurrences
    W = np.zeros((M, D), dtype=np.int32)
    z = np.zeros((M, D), dtype=np.int32)
    
    for m in range(M):
        for d in range(D):
            z[m,d] = np.random.choice(total_K, p=theta[m])
            W[m,d] = np.random.binomial(1, beta[z[m,d], d])
    
    return W, z, beta, theta