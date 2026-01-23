# Latent Factor Allocation (LFA) for Disease Topic Modeling

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A Python package implementing Bayesian inference algorithms for topic modeling in disease data. This package provides both **Mean Field Variational Inference (MFVI)** and **Partially Collapsed Gibbs Sampling (PCGS)** implementations for discovering latent disease topics and their associations.

All algorithms are implemented via NumPy and SciPy.

## What is Topic-Disease Modeling?

Topic modeling for disease data discovers latent patterns in patient diagnoses by identifying groups of diseases that co-occur. We can use topic models to identify disease clusters, comborbidity patterns, and their risk factors. In particular, by probabilistically assigning individuals to disease topics, we can perform a powerful 'Topic-GWAS' to identify genes underlying the progression of disease topics.

## What is LFA?

Latent Factor Allocation (LFA) is a a Bayesian hierarchical model built to infer latent risk profiles for common diseases. It is adapted from Latent Dirichlet Allocation, considering the case of Bernoulli-defined (binary) outcomes; for our purposes, this is to describe disease presence/absence. 

The model assumes there exist a few disease topics that underlie many common diseases. An individual’s risk for each disease is determined by the weights of
all topics.

<div align="center">
  <img src="LFA-DGM.jpg" alt="LFA Directed Graphical Model" width="500">
</div>

<div align="center">
  <em><small><strong>Figure 1:</strong> Plate notation of LFA generative model. M is the number of subjects, D is the number of diseases. All plates (circles) are variables in the generative process, where the plates with shade are the observed variable and plates without shade are unobserved variables to be inferred. θ is the topic weight for all individuals; z is diagnosis-specific topic probability; β is the topic loadings which are Bernoulli probabilities; α is the (non-informative) hyper parameter of the prior distribution of θ.</small></em>
</div>

<br>

For greater detail about the LFA model, see Zhang et al. (2023) [1].


## Available Algorithms in this Package

### Mean Field Variational Inference (MFVI)
A scalable approximate inference method that provides fast inference suitable for large datasets.

MFVI uses variational optimization to approximate the posterior distribution. The Evidence Lower Bound (ELBO) is maximized, and is tracked to monitor convergence.


### Partially Collapsed Gibbs Sampling (PCGS) 
A robust markov-chain monte carlo method that is suitable for small datasets.

PCGS samples from a conditional distribution to exactly approach the posterior distribution. The implementation runs multiple chains which are used to monitor convergence (via Gelman-Rubin statistic) and are averaged once samples are collected.

## Installation

### Prerequisites
- Python 3.8+
- NumPy, SciPy, Matplotlib, Seaborn, pandas

### From Source
```bash
git clone https://github.com/will-marella/LFA
cd LFA
pip install -e .
```

## Quick Start

### Basic MFVI Example
```python
import pandas as pd
from lfa import fit_lfa

# Load your disease data (rows = patients, columns = diseases)
# Values should be 0 (disease absent) or 1 (disease present)
W = pd.read_csv('patient_disease_data.csv', index_col=0)

# Fit LFA model using MFVI (fast, scalable)
result = fit_lfa(
    W,
    num_topics=5,           # Number of disease topics to discover
    algorithm='mfvi',
    max_iterations=1000
)

# View summary of fitted model
print(result.summary())

# See which diseases define each topic
top_diseases = result.get_top_diseases_per_topic(n=10)
for topic, diseases in top_diseases.items():
    print(f"\n{topic}:")
    for disease, loading in diseases:
        print(f"  {disease}: {loading:.3f}")
```

**Or try with synthetic data first:**
```python
from lfa import fit_lfa, simulate_topic_disease_data

W, _, _, _ = simulate_topic_disease_data(seed=42, M=500, D=30, K=3)
result = fit_lfa(W, num_topics=3, algorithm='mfvi')
print(result.summary())
```

### Basic PCGS Example
```python
from lfa import fit_lfa

# For smaller datasets (<500 patients), use PCGS for exact inference
result = fit_lfa(
    W,
    num_topics=5,
    algorithm='pcgs',
    num_chains=3,           # Multiple chains for convergence diagnostics
    max_iterations=2000,
    burn_in=1000
)

# Check convergence (R-hat < 1.1 indicates convergence)
print(f"R-hat: {result.convergence_info['r_hat_overall']:.4f}")
if result.convergence_info['r_hat_overall'] < 1.1:
    print("✓ Converged!")
```


## Data Format

### Input Data (W matrix)
- **Shape**: `(M, D)` where M = subjects, D = diseases
- **Type**: Binary matrix (0/1) indicating presence/absence of each disease
- **Example**:
  ```python
  # For 3 patients and 8 diseases, W might look like:
  W = np.array([
      [0, 1, 0, 1, 0, 1, 0, 0],  # Patient 0: has diseases 1, 3, 5
      [0, 0, 1, 0, 0, 0, 0, 1],  # Patient 1: has diseases 2, 7  
      [1, 1, 0, 0, 1, 0, 0, 0]   # Patient 2: has diseases 0, 1, 4
  ])
  
  # Alternatively, you can set values using:
  W[0, [1, 3, 5]] = 1  # Set multiple diseases for patient 0
  ```

### Model Parameters
- **α (alpha)**: Dirichlet prior for topic proportions `(K,)`
- **β (beta)**: Topic-disease probability matrix `(K, D)`
- **θ (theta)**: Subject-topic proportions `(M, K)`
- **z**: Topic assignments `(M, D)` (discrete) or `(M, D, K)` (probabilistic)

## References

[1] Zhang, Y., Jiang, X., Mentzer, A.J., McVean, G., & Lunter, G. (2023). Topic modeling identifies novel genetic loci associated with multimorbidities in UK Biobank. *Cell Genomics*, 3(8), 100371. https://doi.org/10.1016/j.xgen.2023.100371

## Citation

If you use LFA in your research, please cite:

```bibtex
@software{marella2025lfa,
  author = {Marella, Will},
  title = {LFA: Latent Factor Allocation for Disease Topic Modeling},
  year = {2025},
  url = {https://github.com/will-marella/LFA}
}
```

And the original methodology paper:

```bibtex
@article{zhang2023topic,
  title={Topic modeling identifies novel genetic loci associated with multimorbidities in UK Biobank},
  author={Zhang, Yidong and Jiang, Xilin and Mentzer, Alexander J and McVean, Gil and Lunter, Gerton},
  journal={Cell Genomics},
  volume={3},
  number={8},
  pages={100371},
  year={2023},
  publisher={Elsevier},
  doi={10.1016/j.xgen.2023.100371}
}
```
