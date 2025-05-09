#!/bin/bash
#SBATCH -A INOUYE-SL3-CPU
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH -p icelake
#SBATCH --job-name=mfvi_test
#SBATCH --output=../../results/testing/mfvi_test_%j.out
#SBATCH --error=../../results/testing/mfvi_test_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=wtm24@cam.ac.uk

# Load Python module
module load python/3.9.12/gcc/pdcqf4o5

# Activate virtual environment
source ../../myenv/bin/activate

# Run the experiment
python ../src/scripts/run_mfvi_experiments.py \
    --M 5000 \
    --D 200 \
    --K 10 \
    --topic_prob 0.30 \
    --nontopic_prob 0.01 \
    --max_iterations 100000 \
    --convergence_threshold 1e-6 \
    --seed 42 \
    --results_dir ../../results/testing \
    --experiment_tag "mfvi_initial_test"

echo "Job completed. Results saved in results/testing/" 