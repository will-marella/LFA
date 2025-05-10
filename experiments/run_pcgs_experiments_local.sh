#!/bin/bash

# Activate virtual environment
source myenv/bin/activate

# Create results directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="experiments/pcgs_local_${TIMESTAMP}"
mkdir -p $RESULTS_DIR

# Log file
LOG_FILE="${RESULTS_DIR}/pcgs_experiment.log"

# Experiment tag
EXPERIMENT_TAG="pcgs_local_${TIMESTAMP}"

# Run experiment with smaller parameters for local testing
python src/scripts/run_pcgs_experiments.py \
    --M 40 \
    --D 10 \
    --K 5 \
    --topic_prob 0.3 \
    --nontopic_prob 0.01 \
    --num_chains 2 \
    --max_iterations 1000 \
    --window_size 100 \
    --r_hat_threshold 1.0 \
    --post_convergence_samples 50 \
    --seed 42 \
    --results_dir $RESULTS_DIR \
    --experiment_tag "${EXPERIMENT_TAG}" \
    --log_file $LOG_FILE

echo "Local test completed. Results saved to $RESULTS_DIR" 