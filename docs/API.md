# LFA User API Documentation

This document specifies the clean, user-facing API for the Latent Factor Allocation (LFA) package.

## Main Functions

### `fit_lfa()`

Fit an LFA model to binary disease data.

```python
def fit_lfa(
    W,
    num_topics,
    algorithm='mfvi',
    alpha=None,
    subject_ids=None,
    disease_names=None,
    max_iterations=None,
    convergence_threshold=None,
    num_chains=3,
    verbose=True,
    **algorithm_kwargs
) -> LFAResult
```

**Parameters:**

- **W** : `np.ndarray` or `pd.DataFrame`
  - Binary disease matrix of shape `(M, D)` where M = number of subjects, D = number of diseases
  - Values must be 0 (disease absent) or 1 (disease present)
  - If DataFrame, row/column labels are automatically extracted

- **num_topics** : `int`
  - Number of disease topics to infer (K)
  - Does not include healthy topic (handled internally)

- **algorithm** : `{'mfvi', 'pcgs'}`, default='mfvi'
  - Inference algorithm to use
  - `'mfvi'`: Mean Field Variational Inference (fast, approximate, scalable)
  - `'pcgs'`: Partially Collapsed Gibbs Sampling (slower, exact, for small datasets)

- **alpha** : `np.ndarray` or `float`, optional
  - Dirichlet prior concentration parameter
  - If float: uniform prior `np.ones(K+1) * alpha` is used
  - If array: must have shape `(K+1,)` (includes healthy topic)
  - Default: 0.1 (uniform)

- **subject_ids** : `list` or `np.ndarray`, optional
  - Identifiers for each subject (length M)
  - If W is DataFrame, automatically extracted from index
  - If None, uses integer indices [0, 1, 2, ...]

- **disease_names** : `list` or `np.ndarray`, optional
  - Names for each disease (length D)
  - If W is DataFrame, automatically extracted from columns
  - If None, uses integer indices [0, 1, 2, ...]

- **max_iterations** : `int`, optional
  - Maximum number of iterations
  - Default: 1000 for MFVI, 3000 for PCGS

- **convergence_threshold** : `float`, optional
  - Convergence threshold for stopping
  - Default: 1e-6 for MFVI, uses R-hat for PCGS

- **num_chains** : `int`, default=3
  - Number of chains for PCGS (ignored for MFVI)

- **verbose** : `bool`, default=True
  - Whether to print progress information

- **algorithm_kwargs** : dict
  - Additional algorithm-specific parameters
  - MFVI: `fixed_iterations`, `delta_tail_window`
  - PCGS: `window_size`, `r_hat_threshold`, `post_convergence_samples`

**Returns:**

- **result** : `LFAResult`
  - Object containing fitted parameters and methods for analysis

**Example:**

```python
import numpy as np
from lfa import fit_lfa

# Prepare binary disease data
W = np.array([
    [0, 1, 0, 1, 0],  # Subject 0
    [0, 0, 1, 0, 1],  # Subject 1
    [1, 1, 0, 0, 0],  # Subject 2
])
disease_names = ['Diabetes', 'Hypertension', 'Asthma', 'CAD', 'COPD']

# Fit model
result = fit_lfa(
    W, 
    num_topics=2, 
    algorithm='mfvi',
    disease_names=disease_names
)

# View results
print(result.summary())
```

---

### `select_num_topics()`

Automatically select the optimal number of topics using cross-validation.

```python
def select_num_topics(
    W,
    candidate_topics=None,
    algorithm='mfvi',
    alpha=None,
    n_folds=5,
    metric='bic',
    subject_ids=None,
    disease_names=None,
    verbose=True,
    **algorithm_kwargs
) -> tuple[int, list[dict]]
```

**Parameters:**

- **W** : `np.ndarray` or `pd.DataFrame`
  - Binary disease matrix (M, D)

- **candidate_topics** : `list[int]`, optional
  - List of K values to evaluate
  - Default: [3, 5, 8, 10, 15, 20] (reasonable range for disease data)

- **algorithm** : `str`, default='mfvi'
  - Currently only 'mfvi' is supported for auto-selection
  - PCGS is too slow for cross-validation

- **alpha** : `np.ndarray` or `float`, optional
  - Dirichlet prior (default: 0.1)

- **n_folds** : `int`, default=5
  - Number of cross-validation folds

- **metric** : `{'bic', 'perplexity'}`, default='bic'
  - Metric to use for selection
  - Note: BIC implementation needs validation, use with caution

- **subject_ids** : `list`, optional
  - Subject identifiers

- **disease_names** : `list`, optional
  - Disease names

- **verbose** : `bool`, default=True
  - Print progress

- **algorithm_kwargs** : dict
  - Additional parameters passed to fitting

**Returns:**

- **best_k** : `int`
  - Optimal number of topics based on metric

- **cv_results** : `list[dict]`
  - Cross-validation results for each candidate K
  - Each dict contains: 'K', 'mean_perplexity', 'mean_bic', 'fold_perplexities', 'fold_bics'

**Example:**

```python
from lfa import select_num_topics

# Find optimal K
best_k, cv_results = select_num_topics(
    W, 
    candidate_topics=[3, 5, 8, 10],
    n_folds=5
)

print(f"Best K: {best_k}")
for result in cv_results:
    print(f"K={result['K']}: BIC={result['mean_bic']:.2f}")

# Fit with optimal K
from lfa import fit_lfa
result = fit_lfa(W, num_topics=best_k)
```

---

## LFAResult Class

Object returned by `fit_lfa()` containing fitted parameters and analysis methods.

### Attributes

- **beta** : `np.ndarray`, shape `(K+1, D)`
  - Topic-disease probability matrix
  - `beta[k, d]` = probability of disease d given topic k
  - Includes healthy topic (typically last row)

- **theta** : `np.ndarray`, shape `(M, K+1)`
  - Subject-topic weight matrix
  - `theta[m, k]` = weight of topic k for subject m
  - Rows sum to 1 (Dirichlet constraint)

- **z** : `np.ndarray`
  - Topic assignments for each subject-disease pair
  - MFVI: shape `(M, D, K+1)` probabilistic assignments
  - PCGS: shape `(M, D, K+1)` merged distribution from chains

- **num_topics** : `int`
  - Number of topics (K, excluding healthy topic)

- **num_subjects** : `int`
  - Number of subjects (M)

- **num_diseases** : `int`
  - Number of diseases (D)

- **algorithm** : `str`
  - Algorithm used ('mfvi' or 'pcgs')

- **subject_ids** : `list`
  - Subject identifiers

- **disease_names** : `list`
  - Disease names

- **convergence_info** : `dict`
  - Algorithm-specific convergence information
  - MFVI: `final_elbo`, `elbo_history`, `num_iterations`, `converged`
  - PCGS: `r_hat_beta`, `r_hat_theta`, `num_iterations`, `num_chains`, `converged`

- **metadata** : `dict`
  - Hyperparameters and settings used for fitting

### Methods

#### `summary()`

Print a human-readable summary of the fitted model.

```python
result.summary()
```

#### `to_dict()`

Convert result to dictionary for saving/serialization.

```python
data = result.to_dict()
```

---

## Algorithm-Specific Notes

### MFVI (Mean Field Variational Inference)

**When to use:**
- Large datasets (M > 1000)
- Fast inference needed
- Approximate posteriors acceptable

**Convergence:**
- Monitors ELBO (Evidence Lower Bound)
- Stops when ELBO change < threshold
- Returns `elbo_history` for diagnostics

**Additional kwargs:**
- `fixed_iterations=False`: If True, always run max_iterations
- `delta_tail_window=50`: Window for averaging ELBO changes

### PCGS (Partially Collapsed Gibbs Sampling)

**When to use:**
- Small datasets (M < 500)
- Exact inference needed
- Multiple chains for convergence diagnosis

**Convergence:**
- Monitors Gelman-Rubin R-hat statistic
- Stops when R-hat < threshold (default 1.1)
- Returns per-parameter R-hat values

**Additional kwargs:**
- `num_chains=3`: Number of parallel chains
- `window_size=500`: Window for R-hat calculation
- `r_hat_threshold=1.1`: Convergence threshold
- `post_convergence_samples=50`: Samples to collect after convergence

---

## Notes

- **Healthy topic**: Always included automatically (K+1 total topics)
- **Input validation**: W must be binary (0/1), will raise ValueError otherwise
- **Missing data**: Not currently supported, must impute before fitting
- **Prediction**: MFVI supports inferring theta for new subjects (coming soon)
- **Auto-K selection**: Current BIC implementation needs validation, use with caution

---

## See Also

- [Data Format Guide](DATA_FORMAT.md) - How to prepare input data
- [Potential Output Methods](POTENTIAL_OUTPUT_METHODS.md) - Planned analysis methods
- [Research API](RESEARCH_API.md) - Evaluation with ground truth (for lab members)
