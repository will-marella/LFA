#!/bin/bash
#SBATCH --job-name=mfvi_exp
#SBATCH --output=logs/mfvi_%A_%a.out
#SBATCH --error=logs/mfvi_%A_%a.err
#SBATCH --array=1-10
#SBATCH --time=2:00:00
#SBATCH --mem=8G

# Experiment configuration
M=4000
D=20
K=10
TOPIC_PROB=0.30
NONTOPIC_PROB=0.01
MAX_ITER=2000
CONV_THRESH=1e-6
RESULTS_DIR="/path/to/central/results"
EXP_TAG="baseline_M${M}_K${K}"

# Run experiment with current seed
python src/scripts/run_mfvi_experiments.py \
    --M $M \
    --D $D \
    --K $K \
    --topic_prob $TOPIC_PROB \
    --nontopic_prob $NONTOPIC_PROB \
    --max_iterations $MAX_ITER \
    --convergence_threshold $CONV_THRESH \
    --seed $SLURM_ARRAY_TASK_ID \
    --results_dir $RESULTS_DIR \
    --experiment_tag $EXP_TAG 