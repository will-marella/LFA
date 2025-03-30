import time
import numpy as np

from models.mfvi_model import MFVIModel
from utils.mfvi_monitor import ELBOMonitor
from experiment.get_metrics import align_mfvi_results, compute_mfvi_metrics

def run_mfvi_experiment(W, alpha, num_topics, beta, theta, 
                       max_iterations=1000, convergence_threshold=1e-4):
    """
    Run MFVI experiment with convergence monitoring.
    
    Args:
        W: Observed disease matrix (M x D)
        alpha: Dirichlet prior parameter (K,)
        num_topics: Number of topics K
        beta: True topic-disease matrix for evaluation
        theta: True topic distributions for evaluation
        max_iterations: Maximum number of iterations
        convergence_threshold: ELBO convergence threshold
    """
    start_time = time.time()
    
    # Initialize model and monitor
    model = MFVIModel(W, alpha, num_topics)
    monitor = ELBOMonitor(convergence_threshold, max_iterations)
    
    # Run iterations until convergence
    while True:
        elbo_before, elbo_after, param_changes = model.update_parameters()
        
        if monitor.check_convergence(elbo_before, elbo_after, param_changes):
            break
    
    end_time = time.time()
    run_time = end_time - start_time
    
    # Prepare results
    convergence_stats = monitor.get_convergence_stats()
    
    # Normalize theta for final result
    e_theta_normalized = model.e_theta / np.sum(model.e_theta, axis=1, keepdims=True)
    
    # Create result dictionary
    result = {
        'beta': model.beta,
        'theta': e_theta_normalized,
        'z': model.z,  # Keep the probabilistic assignments
        'elbo_history': convergence_stats['elbo_history'],
        'num_iterations': convergence_stats['num_iterations'],
        'converged': convergence_stats['converged'],
        'final_elbo': convergence_stats['final_elbo']
    }
    
    # Align results with true topics using MFVI-specific alignment
    result = align_mfvi_results(result, beta)
    
    # Compute MFVI-specific metrics
    metrics = compute_mfvi_metrics(result, beta, theta)
    metrics.update({
        'run_time': run_time,
        'parameter_changes': convergence_stats['parameter_changes']
    })
    
    return result, metrics 