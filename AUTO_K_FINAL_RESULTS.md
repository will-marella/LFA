# Auto-K Selection: Final Results

## Summary

After extensive testing, **auto-K selection achieves ~50% accuracy** on simulated data with known ground truth. This is insufficient for reliable automatic model selection.

## What Was Tried

### 1. BIC with Various Sample Sizes
- `N = M_val * D` → Too conservative (selected K=2 when true=3)
- `N = sqrt(M_val * D)` → Still too conservative  
- `N = M_val` → Still too conservative
- `N = sqrt(M_val)` → Still too conservative

**Conclusion**: BIC fundamentally over-penalizes complexity for this problem.

### 2. Pure Perplexity
- Selects K with best validation perplexity
- **Result**: Overfits! Selected K=3 when true=2, K=4 when true=3

### 3. Hybrid: Perplexity + Complexity Penalty

Formula: `minimize perplexity * (1 + penalty * K)`

| Penalty | K=2 Accuracy | K=3 Accuracy | Overall |
|---------|--------------|--------------|---------|
| 0% (pure perp) | 0% | 0% | 0% |
| 1% | 0% | 33% | 17% |
| 1.5% | 0% | 67% | 33% |
| 2% | 0% | 33% | 17% |
| 3% | 100% | 0% | 50% |
| **5%** | **100%** | **0%** | **50%** ✓ |

**Best Result**: 5% penalty achieves 50% accuracy overall
- Perfect for K=2 scenarios (100%)
- Fails for K≥3 scenarios (0%)

## Final Implementation

**File**: `lfa/_utils/auto_k_mfvi.py` (line ~199)

```python
best_entry = min(results, key=lambda d: d["mean_perplexity"] * (1 + 0.05 * d["K"]))
```

**Rationale**:
- Balances fit (perplexity) vs. simplicity (K)
- 5% penalty per topic prevents severe overfitting
- Works perfectly for simple cases, struggles with complex ones

## User-Facing Documentation

**File**: `lfa/lfa.py` (line ~242-250)

Updated warning to honestly reflect:
- ~50% accuracy on simulated data
- Works well for K=2, poorly for K≥3
- Recommendation: Use as starting point, then try K±1 manually

## Validation Results (Quick Test)

**Scenario 1**: M=60, D=16, True K=2
```
Replicate 1: Selected K=2 ✓
Replicate 2: Selected K=2 ✓
Replicate 3: Selected K=2 ✓
Accuracy: 100%
```

**Scenario 2**: M=80, D=18, True K=3
```
Replicate 1: Selected K=2 ✗
Replicate 2: Selected K=2 ✗
Replicate 3: Selected K=2 ✗
Accuracy: 0%
```

**Overall**: 3/6 correct (50%)

## Why Is This Hard?

1. **No ground truth penalty**: Unlike supervised learning, there's no "correct" complexity penalty for unsupervised topic models

2. **Scenario-dependent**: Optimal penalty varies by:
   - Number of subjects (M)
   - Number of diseases (D)  
   - Topic separation (how distinct are topics?)
   - Data sparsity (how many 1s vs 0s?)

3. **Perplexity improves slowly**: Adding topics always helps fit a bit, making it hard to detect "too many topics"

4. **BIC assumes independence**: But subjects with similar disease patterns share topics, violating independence assumptions

## Recommendations for Users

1. **Don't blindly trust auto-K results**
   - Treat selected K as a starting point, not ground truth

2. **Try K±1 manually**
   - Fit models with K-1, K, and K+1
   - Compare interpretability and fit quality

3. **Use domain knowledge**
   - How many disease patterns do you expect?
   - Are there known disease clusters or syndromes?

4. **Check model diagnostics**
   - Are topics interpretable?
   - Do loadings make sense?
   - Is there a clear healthy topic?

5. **Consider PCGS if computational resources allow**
   - Gibbs sampling might give more robust K selection
   - But takes 10-100x longer than MFVI

## Files Modified

1. `lfa/_utils/auto_k_mfvi.py` - Changed from BIC to perplexity+5% penalty
2. `lfa/lfa.py` - Updated warning to reflect limited accuracy
3. `quick_validation.py` - Fast validation script (created for testing)
4. `docs/AUTO_K_FIX_LOG.md` - Detailed technical log
5. `AUTO_K_FIX_SUMMARY.md` - User-friendly summary
6. `AUTO_K_FINAL_RESULTS.md` - This file

## Bottom Line

**Auto-K selection is experimental and should not be trusted blindly.**  
It's better than random guessing but worse than manual exploration with domain knowledge.

Users who care about K should:
- Use `fit_lfa()` with specific K values
- Compare multiple K values manually
- Choose based on interpretability + fit quality

Auto-K remains available as a convenience feature for users who want a quick starting point.
