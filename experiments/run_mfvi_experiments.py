import argparse
import json
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
    parser.add_argument('--output_dir', type=str, default='results/mfvi')
    parser.add_argument('--experiment_name', type=str, required=True, 
                       help='Name for this experiment configuration')
    return parser.parse_args()

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
    
    # Add configuration details to metrics
    metrics['config'] = vars(args)
    
    return metrics

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run experiment
    metrics = run_experiment(args)
    
    # Save results
    output_file = os.path.join(
        args.output_dir, 
        f"{args.experiment_name}_seed{args.seed}.json"
    )
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=2)

if __name__ == '__main__':
    main() 