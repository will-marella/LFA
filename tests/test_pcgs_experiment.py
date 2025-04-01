import os
import numpy as np
from argparse import Namespace

from src.scripts.run_pcgs_experiments import run_experiment, safely_write_results

def test_multiple_configurations():
    # Create test output directory
    test_dir = 'test_output/pcgs_results'
    os.makedirs(test_dir, exist_ok=True)
    results_file = os.path.join(test_dir, "pcgs_results.csv")
    
    # Define two different configurations
    configs = [
        {
            'name': 'small_config',
            'M': 100,
            'D': 20,
            'K': 5,
        },
        {
            'name': 'medium_config',
            'M': 500,
            'D': 24,
            'K': 8,
        }
    ]
    
    # Run experiments for each configuration with two seeds
    for config in configs:
        for seed in [42, 43]:
            # Create arguments
            args = Namespace(
                M=config['M'],
                D=config['D'],
                K=config['K'],
                topic_prob=0.30,
                nontopic_prob=0.01,
                num_chains=2,
                max_iterations=3000,
                window_size=500,
                r_hat_threshold=2.5,
                post_convergence_samples=50,
                seed=seed,
                results_dir=test_dir,
                experiment_tag=config['name']
            )
            
            # Run experiment
            print(f"\nRunning experiment: {config['name']}, seed: {seed}")
            row = run_experiment(args)
            
            # Save results
            safely_write_results(row, results_file)
            
            # Print key metrics
            print(f"Results for {config['name']}, seed {seed}:")
            print(f"Beta MAE: {row['beta_mae']:.4f}")
            print(f"Beta correlation: {row['beta_pearson_corr']:.4f}")
            print(f"Num iterations: {row['num_iterations']}")
            print(f"Run time: {row['run_time']:.2f} seconds")

if __name__ == '__main__':
    test_multiple_configurations() 