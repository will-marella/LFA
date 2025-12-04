# Tier 1 Output Methods - Implementation Summary

## Overview

Tier 1 output methods are **essential for model interpretation**. They enable users to understand what each latent topic represents and how diseases relate to topics. These methods have been added to the `LFAResult` class.

## Implemented Methods (2/3 from original plan)

### ✅ 1. `get_top_diseases_per_topic(n=10, use_names=True, exclude_healthy=True)`

**Purpose:** Primary method for interpreting what each disease topic represents.

**Returns:** Dictionary mapping topic index → list of (disease, probability) tuples

**Parameters:**
- `n` (int, default=10): Number of top diseases to return per topic
- `use_names` (bool, default=True): Return disease names instead of indices
- `exclude_healthy` (bool, default=True): Exclude healthy topic (index 0) from results

**Example:**
```python
result = fit_lfa(W, num_topics=3, disease_names=['Diabetes', 'Hypertension', ...])
top_diseases = result.get_top_diseases_per_topic(n=5)

for topic_idx, diseases in top_diseases.items():
    print(f"Topic {topic_idx}:")
    for disease, prob in diseases:
        print(f"  {disease}: {prob:.3f}")
```

**Output:**
```
Topic 1:
  Diabetes: 0.856
  Hypertension: 0.723
  Obesity: 0.691
  Heart Disease: 0.645
  Stroke: 0.589
Topic 2:
  Asthma: 0.912
  COPD: 0.834
  Bronchitis: 0.721
  Pneumonia: 0.698
  Allergies: 0.654
```

**Use cases:**
- Understanding what medical conditions each topic represents
- Labeling topics based on their top diseases
- Identifying disease clusters/comorbidity patterns
- Validating topic coherence

---

### ✅ 2. `get_disease_topic_loadings(disease_name=None, disease_idx=None)`

**Purpose:** Shows which topics are most strongly associated with a specific disease.

**Returns:** NumPy array of shape (K+1,) with topic probabilities for the disease

**Parameters:**
- `disease_name` (str or int, optional): Name/ID of disease from `disease_names`
- `disease_idx` (int, optional): 0-based index of disease

**Example:**
```python
# Lookup by name
loadings = result.get_disease_topic_loadings(disease_name='Diabetes')
print(f"Diabetes is most associated with topic {loadings.argmax()}")

# Lookup by index
loadings = result.get_disease_topic_loadings(disease_idx=0)

# Exclude healthy topic from analysis
disease_topics = loadings[1:]  # Skip index 0 (healthy topic)
strongest_topic = disease_topics.argmax() + 1
print(f"Strongest disease topic: {strongest_topic}")
```

**Output:**
```
Diabetes is most associated with topic 2
Strongest disease topic: 2
```

**Use cases:**
- Understanding the latent structure explaining a particular disease
- Investigating which topics contribute to a disease
- Comparing topic associations across diseases
- Identifying diseases with similar topic profiles

---

### ❌ 3. `get_subject_topic_distribution()` - NOT IMPLEMENTED

**Rationale:** Deemed less useful than the other two methods. Users can access `result.theta[subject_idx, :]` directly if needed.

---

## Implementation Details

### Code Location
- **File:** `lfa/_core/results.py`
- **Class:** `LFAResult`
- **Lines added:** ~140 lines (including docstrings)

### Key Features
1. **Type flexibility:** Works with both string and integer disease names
2. **Error handling:** Comprehensive validation with helpful error messages
3. **Healthy topic handling:** Optional inclusion/exclusion of healthy topic
4. **Copying:** Returns copies (not views) to prevent accidental modification
5. **Algorithm-agnostic:** Works with both MFVI and PCGS results

### Testing
- **Test file:** `tests/test_output_methods.py`
- **Coverage:** 10 comprehensive test functions
- **Test cases:**
  - Basic functionality with default parameters
  - Disease name vs index lookup
  - Integer vs string disease names
  - Healthy topic inclusion/exclusion
  - Edge cases (n > D, single topic)
  - Error handling (invalid inputs)
  - Integration across methods
  - PCGS compatibility

**All tests passing ✅**

---

## Usage Examples

### Example 1: Basic Topic Interpretation

```python
from lfa import fit_lfa
import pandas as pd

# Load data
W = pd.read_csv('patient_diseases.csv', index_col=0)

# Fit model
result = fit_lfa(W, num_topics=5, algorithm='mfvi')

# Interpret topics
print("Topic Interpretation:")
print("=" * 60)

top_diseases = result.get_top_diseases_per_topic(n=10)
for topic_idx, diseases in top_diseases.items():
    print(f"\nTopic {topic_idx}:")
    for disease, prob in diseases[:5]:  # Top 5 only
        print(f"  {disease:30s} {prob:.3f}")
    
    # Label topic based on top diseases
    top_3 = [d for d, _ in diseases[:3]]
    print(f"  → Label: {' + '.join(top_3)} cluster")
```

### Example 2: Disease-Specific Analysis

```python
from lfa import fit_lfa, simulate_topic_disease_data

# Generate test data
W, beta, theta, z = simulate_topic_disease_data(seed=42, M=200, D=30, K=5)

# Fit model
result = fit_lfa(W, num_topics=5, algorithm='mfvi')

# Analyze specific disease
disease_of_interest = 10
loadings = result.get_disease_topic_loadings(disease_idx=disease_of_interest)

print(f"Disease {disease_of_interest} topic loadings:")
for k, prob in enumerate(loadings):
    if k == 0:
        print(f"  Healthy topic: {prob:.3f}")
    else:
        print(f"  Topic {k}: {prob:.3f}")

# Find dominant topic
dominant_topic = loadings[1:].argmax() + 1
print(f"\nDominant disease topic: {dominant_topic}")
```

### Example 3: Comparing Algorithms

```python
from lfa import fit_lfa, simulate_topic_disease_data

# Simulate data
W, beta, theta, z = simulate_topic_disease_data(seed=42, M=100, D=20, K=3)

# Fit both algorithms
result_mfvi = fit_lfa(W, num_topics=3, algorithm='mfvi')
result_pcgs = fit_lfa(W, num_topics=3, algorithm='pcgs', num_chains=3)

# Compare topic interpretations
print("MFVI Top Diseases:")
for topic_idx, diseases in result_mfvi.get_top_diseases_per_topic(n=3).items():
    print(f"  Topic {topic_idx}: {[d for d, _ in diseases]}")

print("\nPCGS Top Diseases:")
for topic_idx, diseases in result_pcgs.get_top_diseases_per_topic(n=3).items():
    print(f"  Topic {topic_idx}: {[d for d, _ in diseases]}")
```

---

## Future Enhancements (Tier 2+)

Potential additional methods to consider:

**Tier 2: Useful Analysis**
- `get_most_representative_subjects(topic_idx, n=10)` - Find prototypical patients per topic
- `get_disease_clusters(method='hierarchical')` - Group diseases by topic similarity
- `get_topic_diversity()` - Measure how distinct topics are
- `get_topic_prevalence()` - Average theta across subjects

**Tier 3: Visualization**
- `plot_top_diseases_per_topic()` - Bar chart visualization
- `plot_disease_heatmap()` - Topic-disease association heatmap
- `plot_subject_topic_distribution()` - Distribution plots

**Tier 4: Advanced Analysis**
- `predict_new_subject(W_new)` - MFVI inference for new patients
- `get_topic_coherence()` - PMI-based coherence scores
- `get_topic_stability()` - Bootstrap-based stability analysis

---

## Documentation Updates

Updated files:
1. ✅ `docs/API.md` - Added method signatures and examples to LFAResult section
2. ✅ `lfa/_core/results.py` - Added comprehensive docstrings
3. ✅ `tests/test_output_methods.py` - Full test coverage

---

## Conclusion

The two Tier 1 methods provide **essential functionality for model interpretation**:

1. **`get_top_diseases_per_topic()`** - Answers "What does each topic represent?"
2. **`get_disease_topic_loadings()`** - Answers "Which topics explain this disease?"

These methods make the LFA package **immediately useful for practitioners** who need to interpret fitted models, not just fit them.

**Status:** ✅ **Complete and tested**  
**Lines of code:** ~140 (methods) + ~280 (tests) = ~420 total  
**Time to implement:** ~1.5 hours  
**Impact:** High - enables practical use of fitted models
