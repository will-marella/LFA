# LFA: Latent Factor Allocation for Disease Topic Modeling

> **⚠️ DRAFT README** - This is a working draft and will be iteratively improved.

A Python package implementing state-of-the-art Bayesian inference algorithms for topic modeling in disease data. This package provides both **Mean Field Variational Inference (MFVI)** and **Parallel Collapsed Gibbs Sampling (PCGS)** implementations for discovering latent disease topics and their associations.

## 🔬 What is Topic-Disease Modeling?

Topic modeling for disease data discovers latent patterns in patient diagnoses by identifying groups of diseases that co-occur. We can use topic models to identify disease clusters, comborbidity patterns, and their risk factors. In particular, by probabilistically assigning individuals to disease topics, we can perform a powerful 'Topic-GWAS' to identify genes underlying the progression of disease topics.

## Available Algorithms in this Package

### Mean Field Variational Inference (MFVI)
A scalable approximate inference method that:
- Uses variational optimization to approximate the posterior distribution
- Tracks Evidence Lower Bound (ELBO) for convergence monitoring
- Provides fast inference suitable for large datasets
- Offers deterministic results with convergence guarantees

### Parallel Collapsed Gibbs Sampling (PCGS) 
A robust MCMC method that:
- Uses multiple parallel chains for improved convergence diagnostics
- Implements R-hat statistics for convergence assessment
- Provides full posterior samples and uncertainty quantification
- Includes automated convergence detection

## Installation

### Prerequisites
- Python 3.8+
- NumPy, SciPy, Matplotlib, Seaborn, pandas

### From Source
```bash
git clone <repository-url>
cd LFA
pip install -e .
```

### Environment Setup
```bash
# Using conda
conda env create -f environment.yml
conda activate <env-name>

# Or using pip
pip install numpy scipy matplotlib seaborn pandas
```

## Quick Start

### Basic MFVI Example
```python
import numpy as np
from src.experiment.simulation import simulate_topic_disease_data
from src.mfvi_sampler import run_mfvi_experiment

# Generate synthetic disease data
W, z_true, beta_true, theta_true = simulate_topic_disease_data(
    seed=42,
    M=1000,      # Number of patients
    D=20,        # Number of diseases  
    K=3,         # Number of topics
    topic_associated_prob=0.30,
    nontopic_associated_prob=0.01
)

# Run MFVI inference
result, metrics = run_mfvi_experiment(
    W=W,
    alpha=np.ones(4) / 10,  # K+1 for healthy topic
    num_topics=4,
    beta=beta_true,
    theta=theta_true,
    max_iterations=1000
)

print(f"Beta MAE: {metrics['beta_mae']:.4f}")
print(f"Converged in {metrics['num_iterations']} iterations")
```

### Basic PCGS Example
```python
from src.gibbs_sampler import run_cgs_experiment

# Run PCGS inference
result, metrics = run_cgs_experiment(
    W=W,
    alpha=np.ones(4) / 10,
    num_topics=4,
    num_chains=3,           # Multiple chains for convergence
    max_iterations=2000,
    beta=beta_true,
    theta=theta_true,
    r_hat_threshold=1.1     # Convergence threshold
)

print(f"R-hat overall: {metrics['r_hat_overall']:.4f}")
print(f"Converged: {metrics['converged']}")
```

### Running Experiments
Use the provided scripts for systematic experimentation:

```bash
# MFVI experiments
python src/scripts/run_mfvi_experiments.py \
    --M 5000 --D 20 --K 3 \
    --seed 42 \
    --results_dir ./results \
    --experiment_tag "pilot_study"

# PCGS experiments  
python src/scripts/run_pcgs_experiments.py \
    --M 5000 --D 20 --K 3 \
    --num_chains 3 \
    --max_iterations 2000 \
    --seed 42 \
    --results_dir ./results \
    --experiment_tag "pilot_study"
```

## 📊 Data Format

### Input Data (W matrix)
- **Shape**: `(M, D)` where M = subjects, D = diseases
- **Type**: Binary matrix (0/1) indicating presence/absence of each disease
- **Example**:
  ```python
  # Patient 0 has diseases 1, 3, 5
  # Patient 1 has diseases 2, 7
  W[0, [1, 3, 5]] = 1
  W[1, [2, 7]] = 1
  ```

### Model Parameters
- **α (alpha)**: Dirichlet prior for topic proportions `(K,)`
- **β (beta)**: Topic-disease probability matrix `(K, D)`  
- **θ (theta)**: Subject-topic proportions `(M, K)`
- **z**: Topic assignments `(M, D)` (discrete) or `(M, D, K)` (probabilistic)

**Note**: This README is actively being developed. Feedback and contributions are welcome!
