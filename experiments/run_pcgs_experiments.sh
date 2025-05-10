#!/bin/bash
#SBATCH -A INOUYE-SL3-CPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --time=12:00:00
#SBATCH -p icelake
#SBATCH --job-name=pcgs_test
#SBATCH --output=pcgs_test_%j.out
#SBATCH --error=pcgs_test_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wtm24@cam.ac.uk

# Load Python module
module load python/3.9.12/gcc/pdcqf4o5

# Activate virtual environment
source myenv/bin/activate

# Create results directory with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_DIR="experiments/pcgs_${TIMESTAMP}"
mkdir -p $RESULTS_DIR

# Log file
LOG_FILE="${RESULTS_DIR}/pcgs_experiment.log"

# Experiment tag
EXPERIMENT_TAG="pcgs_${TIMESTAMP}"

# Run experiment with specified parameters
python src/scripts/run_pcgs_experiments.py \
    --M 10000 \
    --D 600 \
    --K 10 \
    --topic_prob 0.3 \
    --nontopic_prob 0.01 \
    --num_chains 4 \
    --max_iterations 50000 \
    --window_size 500 \
    --r_hat_threshold 1.0 \
    --post_convergence_samples 500 \
    --seed 42 \
    --results_dir $RESULTS_DIR \
    --experiment_tag "${EXPERIMENT_TAG}" \
    --log_file $LOG_FILE

echo "Experiment completed. Results saved to $RESULTS_DIR" 