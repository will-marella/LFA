import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from experiment.simulation import simulate_topic_disease_data
from mfvi_sampler import run_mfvi_experiment
from experiment.get_metrics import print_mfvi_metrics

if __name__ == '__main__':
    # Simulation parameters
    M = 4000  # Number of subjects
    D = 20  # Number of diseases (without noise)
    num_topics = 10
    seed = 42
    alpha_sim = np.ones(num_topics + 1) / 10

    # Generate synthetic data
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
    num_topics = 11  # Including healthy topic
    alpha = np.ones(num_topics) / 10

    # MFVI details
    max_iterations = 20000
    convergence_threshold = 1e-6

    # Run the experiment
    result, metrics = run_mfvi_experiment(
        W=W,
        alpha=alpha,
        num_topics=num_topics,
        beta=beta,
        theta=theta,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold
    )
    
    # Print results
    print("\nFinal Beta Matrix:")
    print(result['beta'])
    print("\nMetrics:")
    print_mfvi_metrics(metrics)

    # Plotting
    # Beta matrix heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(result['beta'], annot=True, fmt=".2f", cmap='binary', cbar=True)
    plt.title('Heatmap of Beta Matrix from MFVI')
    plt.xlabel('Diseases')
    plt.ylabel('Topics')
    plt.show()

    # ELBO convergence plot
    plt.figure(figsize=(10, 6))
    plt.plot(metrics['elbo_history'])
    plt.title('ELBO Convergence')
    plt.xlabel('Iteration')
    plt.ylabel('ELBO')
    plt.grid(True)
    plt.show()

    # Parameter changes plot
    plt.figure(figsize=(10, 6))
    for param, changes in metrics['parameter_changes'].items():
        plt.plot(changes, label=param)
    plt.title('Parameter Changes Over Iterations')
    plt.xlabel('Iteration')
    plt.ylabel('Average Absolute Change')
    plt.legend()
    plt.grid(True)
    plt.show() 