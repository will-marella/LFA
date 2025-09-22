import argparse
import csv
import os
import numpy as np
import pandas as pd
from pathlib import Path
import time

from src.experiment.simulation import simulate_topic_disease_data
from src.mfvi_sampler import run_mfvi_experiment

def parse_args():
    parser = argparse.ArgumentParser()
    # Data generation parameters
    parser.add_argument('--M', type=int, required=True, help='Number of subjects')
    parser.add_argument('--D', type=int, required=True, help='Number of diseases')
    parser.add_argument('--K', type=int, required=True, help='Number of topics (excluding healthy topic)')
    parser.add_argument('--topic_prob', type=float, default=0.30, help='Topic-associated probability')
    parser.add_argument('--nontopic_prob', type=float, default=0.01, help='Non-topic-associated probability')
    parser.add_argument('--alpha_sim', type=float, default=0.1,
                        help='Dirichlet concentration parameter used for simulation')
    
    # Algorithm parameters
    parser.add_argument('--max_iterations', type=int, default=2000)
    parser.add_argument('--convergence_threshold', type=float, default=1e-6)
    parser.add_argument('--fixed_iterations', action='store_true',
                        help='Disable ELBO-based stopping and run for max_iterations')
    parser.add_argument('--delta_tail_window', type=int, default=50,
                        help='Tail window size for averaging ELBO changes when using fixed iterations')
    
    # Experiment parameters
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--results_dir', type=str, required=True, 
                       help='Directory for results')
    parser.add_argument('--experiment_tag', type=str, required=True,
                       help='Tag to identify this batch of experiments')
    return parser.parse_args()

def flatten_metrics(metrics, args):
    """Flatten metrics into a single row dictionary."""
    row = {
        # Experiment identification
        'experiment_tag': args.experiment_tag,
        'seed': args.seed,
        'timestamp': pd.Timestamp.now(),
        
        # Configuration
        'M': args.M,
        'D': args.D,
        'K': args.K,
        'topic_prob': args.topic_prob,
        'nontopic_prob': args.nontopic_prob,
        'alpha_sim': args.alpha_sim,
        'max_iterations': args.max_iterations,
        'convergence_threshold': args.convergence_threshold,
        'fixed_iterations': args.fixed_iterations,
        'delta_tail_window': args.delta_tail_window,

        # Results
        'num_iterations': metrics['num_iterations'],
        'final_elbo': metrics['final_elbo'],
        'beta_correlation': metrics['beta_correlation'],
        'theta_correlation': metrics['theta_correlation'],
        'beta_mse': metrics['beta_mse'],
        'theta_mse': metrics['theta_mse'],
        'run_time': metrics['run_time'],
        'mean_elbo_delta_tail': metrics.get('mean_elbo_delta_tail'),
        'converged': metrics['converged']
    }
    return row

def safely_write_results(row, results_file):
    """Write results to CSV file with file locking to prevent concurrent access issues."""
    results_dir = os.path.dirname(results_file)
    lock_file = os.path.join(results_dir, ".lock")
    
    # Ensure directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Simple file-based locking
    while os.path.exists(lock_file):
        time.sleep(0.1)
    
    try:
        # Create lock
        with open(lock_file, 'w') as f:
            f.write('locked')
        
        # Read existing results if any
        if os.path.exists(results_file):
            df = pd.read_csv(results_file)
            new_df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        else:
            new_df = pd.DataFrame([row])
        
        # Write back
        new_df.to_csv(results_file, index=False)
            
    finally:
        # Remove lock
        if os.path.exists(lock_file):
            os.remove(lock_file)

def run_experiment(args):
    # Set random seed
    np.random.seed(args.seed)
    
    # Generate synthetic data
    W, z, beta, theta = simulate_topic_disease_data(
        seed=args.seed,
        M=args.M,
        D=args.D,
        K=args.K,
        topic_associated_prob=args.topic_prob,
        nontopic_associated_prob=args.nontopic_prob,
        alpha=np.ones(args.K + 1) * args.alpha_sim,
        include_healthy_topic=True
    )

    # Run MFVI
    result, metrics = run_mfvi_experiment(
        W=W,
        alpha=np.ones(args.K + 1) / 10,
        num_topics=args.K + 1,
        beta=beta,
        theta=theta,
        max_iterations=args.max_iterations,
        convergence_threshold=args.convergence_threshold,
        fixed_iterations=args.fixed_iterations,
        delta_tail_window=args.delta_tail_window
    )

    return flatten_metrics(metrics, args)

def main():
    args = parse_args()
    row = run_experiment(args)
    
    # Define results file path
    results_file = os.path.join(args.results_dir, "mfvi_results.csv")
    
    # Safely write results
    safely_write_results(row, results_file)

if __name__ == '__main__':
    main() 
