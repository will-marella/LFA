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
    
    # Extract the monitor statistics which include R-hat values
    monitor_stats = results.get('monitor_stats', {})
    
    # Get the minimum R-hat value reached
    min_r_hat = float('inf')
    if 'r_hats' in monitor_stats:
        for param, r_hat in monitor_stats['r_hats'].items():
            min_r_hat = min(min_r_hat, r_hat)
    
    # Check if convergence was reached
    converged = monitor_stats.get('max_iterations_reached', False)
    
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
    
    # Add number of iterations (using max across chains)
    # Each chain result has samples collected at different iterations
    metrics['num_iterations'] = max(
        len(chain_result['beta_samples']) 
        for chain_result in chain_results
    )
    
    # Add the minimum R-hat value and convergence status
    metrics['min_r_hat'] = min_r_hat
    metrics['converged'] = converged
    
    # Log the minimum R-hat value
    logger.info(f"Minimum R-hat value reached: {min_r_hat:.4f}")
    logger.info(f"Convergence status: {'Converged' if converged else 'Not converged'}")
    
    return combined_result, metrics



