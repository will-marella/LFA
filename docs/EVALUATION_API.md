# Evaluation API Documentation

## Overview

The LFA evaluation API provides tools for researchers conducting simulation studies to assess parameter recovery against known ground truth. This is useful for validating algorithm performance and comparing different inference methods.

## Key Principle

**Ground truth is only available when you simulate the data yourself.** The evaluation API is designed for simulation studies where you generate data using `simulate_topic_disease_data()` and then test how well the fitting algorithms recover the true parameters.

## Quick Start

```python
from lfa import fit_lfa, simulate_topic_disease_data
from lfa.evaluation import evaluate_result, compare_algorithms

# 1. Simulate data with known ground truth
W, true_beta, true_theta, z = simulate_topic_disease_data(
    seed=42, M=100, D=24, K=3
)

# 2. Fit model
result = fit_lfa(W, num_topics=3, algorithm='mfvi')

# 3. Evaluate against ground truth
metrics = evaluate_result(result, true_beta, true_theta)
print(f"Beta correlation: {metrics['beta_correlation']:.3f}")
print(f"Theta correlation: {metrics['theta_correlation']:.3f}")
```

## API Reference

### `evaluate_result(result, true_beta, true_theta)`

Evaluate fitted result against ground truth parameters.

**Parameters:**
- `result` (LFAResult): Fitted model from `fit_lfa()`
- `true_beta` (np.ndarray): Ground truth topic-disease matrix, shape (K+1, D)
- `true_theta` (np.ndarray): Ground truth subject-topic distributions, shape (M, K+1)

**Returns:**
- `dict`: Metrics dictionary with algorithm-specific keys

**MFVI Metrics:**
- `beta_correlation`: Pearson correlation between aligned beta matrices
- `theta_correlation`: Pearson correlation between aligned theta matrices  
- `beta_mse`: Mean squared error for beta
- `theta_mse`: Mean squared error for theta
- `num_iterations`: Number of iterations run
- `final_elbo`: Final ELBO value
- `converged`: Whether algorithm converged
- `mean_elbo_delta_tail`: Mean ELBO change in tail window (if `fixed_iterations=True`)

**PCGS Metrics:**
- `beta_mae`: Mean absolute error for beta
- `beta_pearson_corr`: Pearson correlation for beta
- `theta_mae`: Mean absolute error for theta
- `theta_pearson_corr`: Pearson correlation for theta
- `num_iterations`: Number of iterations run
- `r_hat_beta`: Gelman-Rubin statistic for beta
- `r_hat_theta`: Gelman-Rubin statistic for theta
- `r_hat_overall`: Maximum R-hat across parameters
- `converged`: Whether chains converged

**Example:**
```python
W, beta, theta, z = simulate_topic_disease_data(seed=42, M=100, D=24, K=3)
result = fit_lfa(W, num_topics=3, algorithm='mfvi')
metrics = evaluate_result(result, beta, theta)

print(f"Recovery quality:")
print(f"  Beta correlation: {metrics['beta_correlation']:.3f}")
print(f"  Theta MSE: {metrics['theta_mse']:.5f}")
print(f"  Converged: {metrics['converged']}")
```

---

### `compare_algorithms(W, num_topics, true_beta, true_theta, algorithms=['mfvi', 'pcgs'], **kwargs)`

Fit multiple algorithms and compare their performance.

**Parameters:**
- `W` (np.ndarray or pd.DataFrame): Binary disease matrix, shape (M, D)
- `num_topics` (int): Number of disease topics (excluding healthy topic)
- `true_beta` (np.ndarray): Ground truth topic-disease matrix, shape (K+1, D)
- `true_theta` (np.ndarray): Ground truth subject-topic distributions, shape (M, K+1)
- `algorithms` (list): Algorithms to compare, e.g. `['mfvi', 'pcgs']`
- `**kwargs`: Algorithm-specific parameters (prefix with algorithm name)

**Algorithm-Specific Parameters:**
Prefix parameters with the algorithm name:
- `mfvi_max_iterations`, `mfvi_convergence_threshold`, `mfvi_verbose`, etc.
- `pcgs_num_chains`, `pcgs_max_iterations`, `pcgs_r_hat_threshold`, `pcgs_verbose`, etc.

**Returns:**
- `pd.DataFrame`: Comparison results with columns for algorithm, runtime, and all metrics

**Example:**
```python
W, beta, theta, z = simulate_topic_disease_data(seed=42, M=100, D=24, K=3)

comparison = compare_algorithms(
    W, num_topics=3,
    true_beta=beta,
    true_theta=theta,
    algorithms=['mfvi', 'pcgs'],
    mfvi_max_iterations=1000,
    mfvi_verbose=False,
    pcgs_num_chains=4,
    pcgs_max_iterations=2000,
    pcgs_verbose=False
)

# View results
print(comparison[['algorithm', 'runtime', 'converged']])
#   algorithm  runtime  converged
# 0      mfvi    12.34       True
# 1      pcgs    45.67       True

# Save for analysis
comparison.to_csv('algorithm_comparison.csv', index=False)

# Compare beta recovery
print(comparison[['algorithm', 'beta_correlation', 'beta_mae']])
```

---

## Simulation Data Format

The `simulate_topic_disease_data()` function returns data in this order:

```python
W, beta, theta, z = simulate_topic_disease_data(seed=42, M=100, D=24, K=3)
```

**Returns:**
- `W` (M, D): Binary disease matrix
- `beta` (K+1, D): True topic-disease probabilities (includes healthy topic)
- `theta` (M, K+1): True subject-topic distributions (includes healthy topic)
- `z` (M, D): True topic assignments

**Note:** The healthy topic is always included (index 0 in beta/theta). The parameter `K` specifies the number of *disease* topics, so total topics = K+1.

---

## Implementation Details

### Topic Alignment

Both MFVI and PCGS results are aligned to ground truth using the **Hungarian algorithm** (Kuhn-Munkres) based on correlation:

1. Compute correlation matrix between estimated and true beta
2. Use `scipy.optimize.linear_sum_assignment` to find optimal topic matching
3. Reorder estimated parameters to match ground truth topics
4. Compute metrics on aligned parameters

This is necessary because topic ordering is arbitrary in unsupervised learning.

### Existing Logic Reuse

The evaluation API is a **minimal wrapper** around existing validated code:

- **MFVI evaluation**: Wraps `align_mfvi_results()` + `compute_mfvi_metrics()` from `lfa._experiment.get_metrics`
- **PCGS evaluation**: Wraps `align_to_simulated_topics()` + `compute_cgs_metrics()` from same module
- **No new metric computation** - all logic existed in legacy code

### File Structure

```
lfa/evaluation/
├── __init__.py          # Exports evaluate_result, compare_algorithms
└── evaluate.py          # Core evaluation functions
```

---

## Example Research Workflow

```python
import numpy as np
import pandas as pd
from lfa import fit_lfa, simulate_topic_disease_data
from lfa.evaluation import compare_algorithms

# Run experiment across multiple parameter settings
results = []

for M in [50, 100, 200]:
    for K in [2, 3, 4]:
        for seed in range(10):  # 10 replicates
            # Simulate
            W, beta, theta, z = simulate_topic_disease_data(
                seed=seed, M=M, D=24, K=K
            )
            
            # Compare algorithms
            comparison = compare_algorithms(
                W, num_topics=K,
                true_beta=beta,
                true_theta=theta,
                algorithms=['mfvi', 'pcgs'],
                mfvi_max_iterations=2000,
                pcgs_num_chains=4
            )
            
            # Add experimental factors
            comparison['M'] = M
            comparison['K'] = K
            comparison['seed'] = seed
            
            results.append(comparison)

# Combine all results
df = pd.concat(results, ignore_index=True)
df.to_csv('full_experiment_results.csv', index=False)

# Analyze
print(df.groupby(['algorithm', 'M', 'K'])['beta_correlation'].mean())
```

---

## Notes

- **Original result not modified**: `evaluate_result()` makes copies before alignment, so the original `LFAResult` object is unchanged
- **PCGS chains**: PCGS results are already aligned across chains (done in `fit_lfa()`), but still need alignment to ground truth for evaluation
- **Different metrics**: MFVI and PCGS use different convergence metrics (ELBO vs R-hat) and recovery metrics (correlation+MSE vs correlation+MAE), reflecting algorithm differences

---

## Bug Fixes Applied

During implementation, we fixed a bug in the existing `align_mfvi_results()` function in `lfa/_experiment/get_metrics.py` where the topic mapping logic was incorrect (missing intermediate `topic_mapping` variable).
