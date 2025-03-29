import time

from models.pcgs import collapsed_gibbs_sampling
from experiment.get_metrics import align_chains, merge_chains, align_to_simulated_topics, compute_cgs_metrics

def run_cgs_experiment(W, alpha, num_topics, num_chains, max_iterations, beta, theta, 
                       window_size, r_hat_threshold=1.1, ess_threshold=400,
                       monitor_params=None, post_convergence_samples=50):
   
    start_time = time.time()
    
    results = collapsed_gibbs_sampling(
        W=W,
        alpha=alpha,
        num_topics=num_topics,
        num_chains=num_chains,
        max_iterations=max_iterations,
        window_size=window_size,
        r_hat_threshold=r_hat_threshold,
        ess_threshold=ess_threshold,
        post_convergence_samples=post_convergence_samples
    )

    end_time = time.time()
    run_time = end_time - start_time
    
    aligned_results = align_chains(results)
    combined_result = merge_chains(aligned_results)
    combined_result = align_to_simulated_topics(combined_result, beta)
    
    metrics = compute_cgs_metrics(combined_result, beta, theta)
    metrics['run_time'] = run_time
    
    return combined_result, metrics



