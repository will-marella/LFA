import argparse
import csv
import os
import numpy as np
import pandas as pd
from pathlib import Path
import time
import logging
import sys
from datetime import datetime

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
    parser.add_argument('--log_file', type=str, default=None,
                       help='Path to log file (if not specified, logs to stdout)')
    return parser.parse_args()

def setup_logging(log_file=None):
    """Set up logging configuration."""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    if log_file:
        # Ensure directory exists
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            datefmt=date_format,
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
    else:
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            datefmt=date_format,
            handlers=[logging.StreamHandler(sys.stdout)]
        )

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
        'run_time': metrics['run_time'],
        'r_hat_beta': metrics.get('r_hat_beta', float('inf')),
        'r_hat_theta': metrics.get('r_hat_theta', float('inf')),
        'r_hat_overall': metrics.get('r_hat_overall', float('inf')),
        'converged': metrics.get('converged', False)
    }
    return row

def safely_write_results(row, results_file):
    """Write results to CSV with file locking."""
    # Ensure directory exists
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
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
        logging.info(f"Results written to {results_file}")
            
    finally:
        # Remove lock
        if os.path.exists(lock_file):
            os.remove(lock_file)

def run_experiment(args):
    # Set random seed
    np.random.seed(args.seed)
    
    # Generate synthetic data
    logging.info(f"Generating synthetic data with M={args.M}, D={args.D}, K={args.K}")
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
    logging.info(f"Starting PCGS with {args.num_chains} chains, max_iterations={args.max_iterations}, r_hat_threshold={args.r_hat_threshold}")
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
        post_convergence_samples=args.post_convergence_samples,
        base_seed=args.seed
    )
    
    # Log final metrics
    logging.info(f"Experiment completed in {metrics['run_time']:.2f} seconds")
    logging.info(f"Number of iterations: {metrics['num_iterations']}")
    logging.info(f"Beta MAE: {metrics['beta_mae']:.4f}")
    logging.info(f"Beta Pearson correlation: {metrics['beta_pearson_corr']:.4f}")
    logging.info(f"Theta MAE: {metrics['theta_mae']:.4f}")
    logging.info(f"Theta Pearson correlation: {metrics['theta_pearson_corr']:.4f}")
    
    if 'r_hat_beta' in metrics:
        logging.info(
            f"R-hat beta: {metrics['r_hat_beta']:.4f}, theta: {metrics['r_hat_theta']:.4f}, overall: {metrics['r_hat_overall']:.4f}"
        )
    
    return flatten_metrics(metrics, args)

def main():
    args = parse_args()
    
    # Set up logging
    if args.log_file:
        # If log_file is specified but doesn't include a directory, put it in results_dir
        if not os.path.dirname(args.log_file):
            args.log_file = os.path.join(args.results_dir, args.log_file)
    else:
        # Default log file in results_dir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log_file = os.path.join(args.results_dir, f"pcgs_log_{timestamp}.log")
    
    setup_logging(args.log_file)
    logging.info(f"Starting PCGS experiment with tag: {args.experiment_tag}")
    logging.info(f"Arguments: {vars(args)}")
    
    try:
        row = run_experiment(args)
        
        # Define results file path
        results_file = os.path.join(args.results_dir, "pcgs_results.csv")
        
        # Safely write results
        safely_write_results(row, results_file)
        
        logging.info("Experiment completed successfully")
    except Exception as e:
        logging.error(f"Experiment failed: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main() 