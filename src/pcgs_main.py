import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from experiment.simulation import simulate_topic_disease_data
from gibbs_sampler import run_cgs_experiment
from experiment.get_metrics import print_metrics



# Main block
if __name__ == '__main__':
    # Simulation parameters
    M = 40 # Number of subjects
    D = 20   # Number of diseases (without noise)
    num_topics = 10
    seed = 42
    alpha_sim = np.ones(num_topics + 1) / 10

    W, z, beta, theta = simulate_topic_disease_data(
        seed=seed, 
        M=M, 
        D=D, 
        K=num_topics,
        topic_associated_prob=0.30,
        nontopic_associated_prob=0.01,
        alpha=alpha_sim,
        include_healthy_topic=True
    )

    # Initialization Parameters
    num_topics = 11
    alpha = np.ones(num_topics) / 10

    # PCGS details
    num_chains = 2
    max_iterations = 2000
    window_size = 500
    monitor_params = ['beta', 'theta']
    post_convergence_samples = 100

    # Run the experiment
    combined_result, metrics = run_cgs_experiment(
        W=W,
        alpha=alpha,
        num_topics=num_topics,
        num_chains=num_chains,
        max_iterations=max_iterations,
        beta=beta,
        theta=theta,
        window_size=window_size,
        r_hat_threshold=1.0,
        calculate_ess=False,
        monitor_params=monitor_params,
        post_convergence_samples=post_convergence_samples
    )
    
    # Print results
    print(combined_result['beta'])
    print_metrics(metrics)

    # Plotting
    plt.figure(figsize=(10, 8))
    sns.heatmap(combined_result['beta'], annot=True, fmt=".2f", cmap='binary', cbar=True)
    plt.title('Heatmap of Beta Matrix from CGS')
    plt.xlabel('Diseases')
    plt.ylabel('Topics')
    plt.show()
