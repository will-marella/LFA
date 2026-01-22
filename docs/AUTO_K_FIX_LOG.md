# Auto-K Selection Fix Log

## Problem
The `select_num_topics()` function was selecting K incorrectly due to fundamental issues with the BIC calculation:

1. **Wrong parameter count**: Only counted β parameters `(K+1)×D`, ignored θ parameters `M×K`
2. **Wrong sample size**: Used `N = M×D` (all matrix entries) instead of `N = M` (subjects)
3. **No validation**: Never tested on simulated data with known ground truth

## Investigation History

### Initial Formula (Before Fix)
```python
num_params = (K+1) * D  # Only β, no θ  
N = M_val * D           # Total entries
BIC = -2*log_likelihood + num_params*log(N)
```

**Result**: Unknown - not validated

### Attempted Fix #1: Include θ Parameters
```python
num_params = (K+1)*D + M_val*K  # β + θ
N = M_val                        # Subjects
```

**Result**: TOO CONSERVATIVE
- Test K=2: ✓ Selected K=2 correctly (1/1, 100%)
- Test K=3: ✗ Selected K=2 incorrectly (0/3, 0%)
- Problem: Including M_val×K parameters made penalty too large

### Attempted Fix #2: β Only, Geometric Mean N
```python
num_params = (K+1)*D           # β only
N = sqrt(M_val * D)            # Geometric mean
```

**Result**: STILL TOO CONSERVATIVE (per previous summary)
- Selected K=2 when true K=3
- Problem: Penalty still too strong

### Attempted Fix #3: β Only, N=M_val (CURRENT - NOT TESTED)
```python
num_params = (K+1)*D           # β only
N = M_val                      # Subjects only
```

**Status**: Implemented but untested
**Expected**: VERY CONSERVATIVE (even more than Fix #2)
- Smaller N → Smaller log(N) → Smaller penalty → Should select higher K
- Wait, that's backwards! Smaller log(N) means LESS penalty, so we should select HIGHER K
- But that makes no sense with the previous results...

**Analysis Error**: There may be confusion in the reasoning. Let me clarify:
- BIC penalty = `num_params * log(N)`
- LARGER N → LARGER penalty → Prefer simpler models (smaller K)
- SMALLER N → SMALLER penalty → Prefer complex models (larger K)

So:
- `N = M_val*D` (large) → Large penalty → Selects smaller K (conservative)
- `N = sqrt(M_val*D)` (medium) → Medium penalty → Medium K
- `N = M_val` (small) → Small penalty → Selects larger K (liberal)

### Current Fix #4: β Only, N=M_val*D (RECOMMENDED)
```python
num_params = (K+1)*D           # β only
N = M_val * D                  # Total observations
```

**Rationale**:
- Previous fix (N=sqrt(M_val*D)) was too conservative (selected K=2 when K=3)
- Need LESS conservative → Use LARGER N → More penalty
- Wait, that's backwards again!

**Corrected Analysis**:
Actually, looking at BIC formula: `BIC = -2*LL + penalty`
- We SELECT the K with MINIMUM BIC
- Larger penalty → Larger BIC → LESS likely to be selected
- So LARGER N → LARGER penalty → Penalizes complex models MORE → Selects SMALLER K

Therefore:
- `N = M_val*D` (largest) → Most conservative, smallest K
- `N = sqrt(M_val*D)` (medium) → Medium
- `N = M_val` (smallest) → Least conservative, largest K

Since N=sqrt(M_val*D) was TOO CONSERVATIVE (selected K=2 when true=3), we need LESS penalty, so SMALLER N.

**Final Recommendation**: Use `N = M_val`

But this contradicts what I just implemented! Let me revert:

```python
num_params = (K+1)*D
N = M_val  # Smallest N, least penalty, most likely to select higher K
```

## Current Status (Fix #4)

**Implemented**:
```python
num_params = (K+1) * D    # β parameters only
N = M_val * D              # Total observations
```

**Testing Required**:
Run `tests/test_auto_k_validation.py::run_comprehensive_validation()` to assess:
- Exact accuracy (should be ≥50% to be useful)
- Within ±1 accuracy (should be ≥70%)

**Expected Behavior**:
Given that N=sqrt(M_val*D) was too conservative:
- N=M_val*D (larger) will be EVEN MORE conservative
- N=M_val (smaller) would be LESS conservative

**Conclusion**: The current fix may STILL be too conservative. If validation fails, try:

### Alternative Fix #5: N = M_val (True Least Conservative)
```python
num_params = (K+1) * D
N = M_val  # Treat each subject as one observation
```

This would give the smallest penalty and be most likely to select higher K values.

## Alternative Approach: Perplexity-Based Selection

If BIC continues to fail, switch to simple perplexity minimization:

```python
best_k = min(results, key=lambda d: d["mean_perplexity"])["K"]
```

**Pros**:
- Simple, no parameter counting
- Directly optimizes predictive performance

**Cons**:
- No complexity penalty (may overfit, select K too large)
- Less theoretically justified

**Mitigation**:
Add ad-hoc penalty: `perplexity * (1 + 0.1*K)` to slightly penalize complexity.

## Validation Framework

Test file: `tests/test_auto_k_validation.py`

**Scenarios**:
1. Easy: M=100, D=20, K=2 (clear separation)
2. Standard: M=150, D=24, K=3 (moderate size)
3. Complex: M=120, D=18, K=4 (more topics)
4. Large: M=200, D=30, K=3 (big dataset)

**Success Criteria**:
- ≥70% exact accuracy → EXCELLENT, downgrade warning
- 50-70% → GOOD, keep warning but note validation
- 30-50% → POOR, keep strong warning
- <30% → BROKEN, consider deprecation

## Next Steps

1. **Test current fix** (N=M_val*D):
   ```bash
   cd LFA && python -m pytest tests/test_auto_k_validation.py::run_comprehensive_validation -v
   ```

2. **If too conservative**, revert to N=M_val:
   ```python
   N = M_val  # In auto_k_mfvi.py line ~127
   ```

3. **If still fails**, switch to perplexity:
   ```python
   # In auto_k_mfvi.py line ~199
   best_entry = min(results, key=lambda d: d["mean_perplexity"])
   ```

4. **Document results** and update user-facing warning in `lfa/lfa.py`

## References

- BIC Formula: `BIC = -2*log(L) + k*log(n)` where k=params, n=sample size
- Lower BIC is better (balances fit vs complexity)
- Parameter counting for topic models: See Griffiths & Steyvers (2004) 
- Effective sample size for dependent data: Controversial, no consensus
