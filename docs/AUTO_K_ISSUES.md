# Auto-K Selection Issues

## Current Status: BROKEN ❌

The `select_num_topics()` function uses BIC (Bayesian Information Criterion) for model selection, but there are **fundamental issues** with the current implementation.

## Location

- **Main API**: `lfa/lfa.py::select_num_topics()`
- **Implementation**: `lfa/_utils/auto_k_mfvi.py::select_k_mfvi()`
- **BIC Calculation**: Line 118 in `auto_k_mfvi.py`

## Issues Identified

### 1. **Incorrect Parameter Count** (Line 84)

```python
num_params = (K + 1) * D  # β parameters only
```

**Problem**: Only counts β (topic-disease) parameters, ignores θ (subject-topic) parameters.

**Should be**:
```python
# β parameters: (K+1) × D
# θ parameters: M × K (K not K+1 due to Dirichlet constraint)
num_params = (K + 1) * D + M * K
```

**Impact**: Severely under-penalizes complex models, favoring higher K.

---

### 2. **Questionable Log-Likelihood from Perplexity** (Lines 117-118)

```python
N_val = W_val.size  # Total number of entries (M_val × D)
log_likelihood = -N_val * np.log(perp)
bic = -2 * log_likelihood + num_params * np.log(N_val)
```

**Problems**:
a) **Sample size ambiguity**: Should `N` be:
   - `M_val × D` (number of data points)? 
   - `M_val` (number of subjects)?
   - Something else?

b) **Perplexity → Log-Likelihood conversion**: The relationship between perplexity and log-likelihood for **binary** data might not be straightforward. Standard perplexity assumes categorical distributions.

c) **Independence assumption**: BIC assumes i.i.d. observations, but in LFA:
   - Diseases within a subject are NOT independent (they share θ_m)
   - Effective sample size might be `M` (subjects), not `M × D` (entries)

---

### 3. **Returns Wrong Metric** (Line 182)

```python
best_entry = min(results, key=lambda d: d["mean_bic"])
```

**Problem**: Selects based on BIC, but the function docstring (line 151) says:
> "K value with the lowest mean **perplexity**"

**Inconsistency**: The code and documentation disagree on selection criterion.

---

### 4. **No Direct ELBO Comparison**

The code trains full MFVI models on each fold and computes ELBO during training, but **never uses ELBO for model selection**. 

**Alternative**: Could use held-out ELBO directly rather than converting to perplexity → log-likelihood → BIC.

---

## Why It's Broken in Practice

Given data with TRUE K=3:

1. **Under-counting parameters** favors higher K
2. **BIC penalty too weak** → overfitting not penalized enough
3. **Might select K=5 or K=10** instead of K=3

The current implementation is **not validated** against simulated data where true K is known.

---

## Potential Fixes

### Option A: Fix BIC Calculation

```python
def _cv_score_single_K_FIXED(...):
    # ... existing training code ...
    
    M_val = W_val.shape[0]
    D = W_val.shape[1]
    
    # Correct parameter count
    num_params = (K + 1) * D + M_val * K  # β + θ parameters
    
    # Compute held-out log-likelihood directly
    # (Rather than via perplexity)
    log_probs = W_val * np.log(P_pred + 1e-10) + (1 - W_val) * np.log(1 - P_pred + 1e-10)
    log_likelihood = log_probs.sum()
    
    # BIC with effective sample size = number of subjects
    bic = -2 * log_likelihood + num_params * np.log(M_val)
    
    # Also compute perplexity for reporting
    perp = compute_perplexity(W_val, P_pred)
    
    fold_bics.append(float(bic))
    fold_perps.append(float(perp))
```

**Changes:**
1. ✅ Include θ parameters in count
2. ✅ Compute log-likelihood directly from predictions
3. ✅ Use `M_val` (subjects) as sample size, not `M_val × D`
4. ✅ Keep perplexity separate for reporting

---

### Option B: Use Cross-Validated ELBO

```python
def _cv_score_single_K_ELBO(...):
    # ... existing training code ...
    
    # Compute held-out ELBO for validation subjects
    # This requires inferring θ for validation subjects
    # and computing ELBO term for them
    
    val_elbos = []
    for i, row in enumerate(W_val):
        theta_exp, _ = infer_theta_single_row(...)
        # Compute ELBO contribution for this subject
        elbo_i = compute_subject_elbo(row, theta_exp, beta_hat, alpha)
        val_elbos.append(elbo_i)
    
    mean_val_elbo = np.mean(val_elbos)
    fold_elbos.append(mean_val_elbo)
    
# Select K with highest mean validation ELBO
best_k = max(results, key=lambda d: d["mean_val_elbo"])
```

**Advantage:** Uses the model's native objective (ELBO) rather than converting through perplexity.

---

### Option C: Use Perplexity Directly

Simplest fix: Just select based on perplexity (already computed correctly).

```python
best_entry = min(results, key=lambda d: d["mean_perplexity"])
```

**Advantage:** 
- Perplexity is a proper scoring rule for prediction
- Already implemented correctly
- No BIC confusion

**Disadvantage:**
- No parameter penalty → might still overfit
- Not a formal model selection criterion

---

### Option D: Abandon Auto-K

Document that auto-K is unreliable and users should:

1. **Try multiple K values** manually
2. **Use domain knowledge** to set K
3. **Inspect topic interpretability** (coherence, distinctness)
4. **Validate on held-out data** if available

Update docs to provide **guidance on reasonable K ranges**:
- Small datasets (M < 100): K ∈ {2, 3, 4, 5}
- Medium datasets (M = 100-500): K ∈ {3, 5, 8, 10}
- Large datasets (M > 500): K ∈ {5, 10, 15, 20}

---

## Recommendation

**For now**: Keep Option D (document as experimental, guide users to manual selection)

**For future**: Implement Option A (fixed BIC) and validate on simulations

---

## Testing Strategy (if fixing)

```python
# Generate data with known K
for true_k in [2, 3, 4, 5]:
    for seed in range(10):  # Multiple replicates
        W, beta, theta, z = simulate_topic_disease_data(
            seed=seed, M=200, D=30, K=true_k
        )
        
        # Test auto-K
        selected_k, results = select_num_topics(
            W, candidate_topics=[2, 3, 4, 5, 6, 7]
        )
        
        # Record accuracy
        correct = (selected_k == true_k)
```

**Acceptance criteria:**
- Should select correct K at least 70% of the time
- Should never select K > true_K + 2
- BIC should monotonically increase for K > true_K

---

## References

- **BIC Formula**: Schwarz, G. (1978). "Estimating the dimension of a model"
- **LDA Model Selection**: Griffiths & Steyvers (2004) - use perplexity
- **Topic Model Selection**: Wallach et al. (2009) - discusses evaluation metrics

---

## Current Warning

The code currently shows this warning (line 329 in `lfa.py`):

```python
warnings.warn(
    "BIC-based selection is experimental and needs validation. "
    "Consider manually trying different K values.",
    UserWarning
)
```

**This warning is appropriate and should remain until BIC is validated.**
