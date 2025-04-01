import argparse
import csv
import os
import numpy as np
import pandas as pd
from pathlib import Path
import time

from src.experiment.simulation import simulate_topic_disease_data
from src.gibbs_sampler import run_cgs_experiment

def parse_args():
    parser = argparse.ArgumentParser()
    # Data generation parameters
    parser.add_argument('--M', type=int, required=True, help='Number of subjects')
    parser.add_argument('--D', type=int, required=True, help='Number of diseases')
    parser.add_argument('--K', type=int, required=True, help='Number of topics (excluding healthy topic)')
    parser.add_argument('--topic_prob', type=float, default=0.30, help='Topic-associated probability')
    parser.add_argument('--nontopic_prob', type=float, default=0.01, help='Non-topic-associated probability')
    
    # Algorithm parameters
    parser.add_argument('--num_chains', type=int, default=2, help='Number of parallel chains')
    parser.add_argument('--max_iterations', type=int, default=2000)
    parser.add_argument('--window_size', type=int, default=500, help='Window size for convergence check')
    parser.add_argument('--r_hat_threshold', type=float, default=1.1, help='R-hat convergence threshold')
    parser.add_argument('--post_convergence_samples', type=int, default=100)
    
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
        'num_chains': args.num_chains,
        'max_iterations': args.max_iterations,
        'window_size': args.window_size,
        'r_hat_threshold': args.r_hat_threshold,
        'post_convergence_samples': args.post_convergence_samples,
        
        # Results
        'num_iterations': metrics['num_iterations'],
        'beta_mae': metrics['beta_mae'],
        'beta_pearson_corr': metrics['beta_pearson_corr'],
        'theta_mae': metrics['theta_mae'],
        'theta_pearson_corr': metrics['theta_pearson_corr'],
        'run_time': metrics['run_time']
    }
    return row

def safely_write_results(row, results_file):
    """Write results to CSV with file locking."""
    lock_file = results_file + '.lock'
    
    # Try to acquire lock
    while os.path.exists(lock_file):
        time.sleep(0.1)
    
    try:
        # Create lock
        Path(lock_file).touch()
        
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
        alpha=np.ones(args.K + 1) / 10,
        include_healthy_topic=True
    )
    
    # Run PCGS
    result, metrics = run_cgs_experiment(
        W=W,
        alpha=np.ones(args.K + 1) / 10,
        num_topics=args.K + 1,  # +1 for healthy topic
        num_chains=args.num_chains,
        max_iterations=args.max_iterations,
        beta=beta,
        theta=theta,
        window_size=args.window_size,
        r_hat_threshold=args.r_hat_threshold,
        post_convergence_samples=args.post_convergence_samples
    )
    
    return flatten_metrics(metrics, args)

def main():
    args = parse_args()
    row = run_experiment(args)
    
    # Define results file path
    results_file = os.path.join(args.results_dir, "pcgs_results.csv")
    
    # Safely write results
    safely_write_results(row, results_file)

if __name__ == '__main__':
    main() 