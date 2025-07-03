import time
import logging

from src.models.pcgs import collapsed_gibbs_sampling
from src.experiment.get_metrics import align_chains, merge_chains, align_to_simulated_topics, compute_cgs_metrics

def run_cgs_experiment(W, alpha, num_topics, num_chains, max_iterations, beta, theta, 
                       window_size, r_hat_threshold=1.1, calculate_ess=False, ess_threshold=400,
                       monitor_params=None, post_convergence_samples=50):
   
    start_time = time.time()
    
    # Create a custom logger for this experiment
    logger = logging.getLogger('pcgs_experiment')
    
    # Run the Gibbs sampling
    results = collapsed_gibbs_sampling(
        W=W,
        alpha=alpha,
        num_topics=num_topics,
        num_chains=num_chains,
        max_iterations=max_iterations,
        window_size=window_size,
        r_hat_threshold=r_hat_threshold,
        calculate_ess=calculate_ess,
        ess_threshold=ess_threshold,
        post_convergence_samples=post_convergence_samples
    )

    end_time = time.time()
    run_time = end_time - start_time
    
    # Extract monitor statistics which include parameter-specific R-hat values
    monitor_stats = results.get('monitor_stats', {})

    r_hat_beta = monitor_stats.get('r_hats', {}).get('beta', float('inf'))
    r_hat_theta = monitor_stats.get('r_hats', {}).get('theta', float('inf'))
    # Define an overall criterion (worst of the two)
    r_hat_overall = max(r_hat_beta, r_hat_theta)
    
    # Determine convergence based on R-hat threshold
    converged = r_hat_overall < r_hat_threshold
    
    # Process the chain results
    chain_results = results.get('chain_results', [])
    if not chain_results:
        logger.warning("No chain results found in the output")
        return None, {'run_time': run_time, 'error': 'No chain results'}
    
    aligned_results = align_chains(chain_results)
    combined_result = merge_chains(aligned_results)
    combined_result = align_to_simulated_topics(combined_result, beta)
    
    metrics = compute_cgs_metrics(combined_result, beta, theta)
    metrics['run_time'] = run_time
    
    # Record number of iterations actually performed (max across chains)
    if 'chain_iterations' in monitor_stats:
        metrics['num_iterations'] = max(monitor_stats['chain_iterations'].values())
    else:
        metrics['num_iterations'] = None  # Fallback when information missing
    
    # Also record how many post-convergence samples were collected per chain
    metrics['num_samples_collected'] = max(
        len(chain_result['beta_samples'])
        for chain_result in chain_results
    )
    
    # Add R-hat values and convergence status
    metrics['r_hat_beta'] = r_hat_beta
    metrics['r_hat_theta'] = r_hat_theta
    metrics['r_hat_overall'] = r_hat_overall
    metrics['converged'] = converged
    
    # Log the R-hat values
    logger.info(
        f"R-hat values – beta: {r_hat_beta:.4f}, theta: {r_hat_theta:.4f}, overall(max): {r_hat_overall:.4f}"
    )
    logger.info(f"Convergence status: {'Converged' if converged else 'Not converged'}")
    
    return combined_result, metrics



