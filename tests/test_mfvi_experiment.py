import os
import numpy as np
import csv

from src.scripts.run_mfvi_experiments import run_experiment, main
from argparse import Namespace

def test_single_experiment():
    # Create test output directory
    os.makedirs('test_output', exist_ok=True)
    
    # Create test arguments
    args = Namespace(
        M=100,  # Small number of subjects for quick testing
        D=20,
        K=5,
        topic_prob=0.30,
        nontopic_prob=0.01,
        max_iterations=1000,
        convergence_threshold=1e-6,
        seed=42,
        output_file='test_output/test_mfvi_results.csv'
    )
    
    # Run experiment
    row = run_experiment(args)
    
    # Write results
    is_new_file = not os.path.exists(args.output_file)
    
    with open(args.output_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if is_new_file:
            writer.writeheader()
        writer.writerow(row)
    
    # Print results to verify
    print("\nExperiment Results:")
    for key, value in row.items():
        print(f"{key}: {value}")

if __name__ == '__main__':
    test_single_experiment() 