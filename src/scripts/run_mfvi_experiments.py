import argparse
import csv
import os
import numpy as np

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
    
    # Algorithm parameters
    parser.add_argument('--max_iterations', type=int, default=2000)
    parser.add_argument('--convergence_threshold', type=float, default=1e-6)
    
    # Experiment parameters
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output_file', type=str, required=True, 
                       help='Path to output CSV file')
    return parser.parse_args()

def flatten_metrics(metrics, args):
    """Flatten metrics into a single row dictionary."""
    row = {
        # Configuration
        'seed': args.seed,
        'M': args.M,
        'D': args.D,
        'K': args.K,
        'topic_prob': args.topic_prob,
        'nontopic_prob': args.nontopic_prob,
        'max_iterations': args.max_iterations,
        'convergence_threshold': args.convergence_threshold,
        
        # Results
        'num_iterations': metrics['num_iterations'],
        'final_elbo': metrics['final_elbo'],
        'beta_correlation': metrics['beta_correlation'],
        'theta_correlation': metrics['theta_correlation'],
        'beta_mse': metrics['beta_mse'],
        'theta_mse': metrics['theta_mse'],
        'run_time': metrics['run_time'],
        'converged': metrics['num_iterations'] < args.max_iterations  # Determine convergence from iterations
    }
    return row

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
    
    # Run MFVI
    result, metrics = run_mfvi_experiment(
        W=W,
        alpha=np.ones(args.K + 1) / 10,
        num_topics=args.K + 1,  # +1 for healthy topic
        beta=beta,
        theta=theta,
        max_iterations=args.max_iterations,
        convergence_threshold=args.convergence_threshold
    )
    
    return flatten_metrics(metrics, args)

def main():
    args = parse_args()
    row = run_experiment(args)
    
    # Write results to CSV
    is_new_file = not os.path.exists(args.output_file)
    
    with open(args.output_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if is_new_file:
            writer.writeheader()
        writer.writerow(row)

if __name__ == '__main__':
    main() 