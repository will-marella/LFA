# Potential Output Methods for LFAResult

This document catalogs potential analysis methods to add to the `LFAResult` class in future phases. These are inspired by popular LDA implementations (scikit-learn, gensim) adapted for binary disease data.

## Priority: High (Essential for Interpretation)

### `get_top_diseases_per_topic(n=10, use_names=True)`
Returns the top N diseases most associated with each topic.

**Returns:** Dictionary mapping topic index to list of (disease, probability) tuples

**Example:**
```python
top_diseases = result.get_top_diseases_per_topic(n=5)
for topic_idx, diseases in top_diseases.items():
    print(f"Topic {topic_idx}:")
    for disease, prob in diseases:
        print(f"  {disease}: {prob:.3f}")
```

**Use case:** Primary method for interpreting what each topic represents

---

### `get_subject_topic_distribution(subject_id=None, subject_idx=None)`
Get the topic distribution (theta) for a specific subject.

**Returns:** Array of shape `(K+1,)` with topic weights

**Example:**
```python
# By ID
theta_patient = result.get_subject_topic_distribution(subject_id='P001')

# By index
theta_patient = result.get_subject_topic_distribution(subject_idx=0)
```

**Use case:** Understand an individual patient's disease risk profile

---

### `get_disease_topic_loadings(disease_name=None, disease_idx=None)`
Get the topic loadings (beta) for a specific disease.

**Returns:** Array of shape `(K+1,)` showing which topics associate with this disease

**Example:**
```python
loadings = result.get_disease_topic_loadings(disease_name='Diabetes')
print(f"Diabetes is most associated with topic {loadings.argmax()}")
```

**Use case:** Understand which disease topics explain a particular condition

---

## Priority: Medium (Useful Analysis)

### `get_most_representative_subjects(topic_idx, n=10, use_ids=True)`
Find subjects most strongly associated with a given topic.

**Returns:** List of (subject_id, theta_weight) tuples

**Example:**
```python
top_subjects = result.get_most_representative_subjects(topic_idx=2, n=10)
for subject, weight in top_subjects:
    print(f"{subject}: {weight:.3f}")
```

**Use case:** Find "prototypical" patients for each disease topic; useful for validation

---

### `get_disease_clusters(method='hierarchical', threshold=0.5)`
Group diseases by topic similarity.

**Returns:** Dictionary mapping cluster_id to list of diseases

**Example:**
```python
clusters = result.get_disease_clusters()
for cluster_id, diseases in clusters.items():
    print(f"Cluster {cluster_id}: {', '.join(diseases)}")
```

**Use case:** Discover disease comorbidity patterns

---

### `get_topic_diversity()`
Measure how distinct topics are from each other (entropy-based or correlation-based).

**Returns:** Float between 0 (topics identical) and 1 (topics maximally distinct)

**Example:**
```python
diversity = result.get_topic_diversity()
print(f"Topic diversity score: {diversity:.3f}")
```

**Use case:** Assess model quality; low diversity suggests redundant topics

---

### `get_subject_primary_topic(subject_id=None, subject_idx=None)`
Get the dominant topic for a subject.

**Returns:** Tuple of (topic_idx, weight)

**Example:**
```python
topic, weight = result.get_subject_primary_topic(subject_id='P001')
print(f"Patient P001's primary topic: {topic} (weight: {weight:.3f})")
```

**Use case:** Simple categorization of patients

---

## Priority: Low (Nice to Have)

### `compare_subjects(subject_id_1, subject_id_2)`
Compare topic distributions of two subjects.

**Returns:** Dictionary with similarity metrics (cosine, KL divergence, etc.)

**Example:**
```python
comparison = result.compare_subjects('P001', 'P002')
print(f"Cosine similarity: {comparison['cosine']:.3f}")
print(f"KL divergence: {comparison['kl_divergence']:.3f}")
```

**Use case:** Find similar patients; understand disease profile differences

---

### `get_disease_cooccurrence_matrix(topic_weighted=True)`
Compute disease co-occurrence, optionally weighted by topic structure.

**Returns:** DataFrame or array `(D, D)` with co-occurrence scores

**Example:**
```python
cooccurrence = result.get_disease_cooccurrence_matrix()
# Shows which diseases frequently appear together
```

**Use case:** Network analysis of disease relationships

---

### `get_topic_prevalence()`
Calculate the average weight of each topic across all subjects.

**Returns:** Array of shape `(K+1,)` with prevalence scores

**Example:**
```python
prevalence = result.get_topic_prevalence()
for i, prev in enumerate(prevalence):
    print(f"Topic {i}: {prev:.2%} of population")
```

**Use case:** Understand population-level disease burden by topic

---

### `filter_by_topic(topic_idx, threshold=0.5)`
Get subjects strongly associated with a specific topic.

**Returns:** List of subject IDs

**Example:**
```python
topic_2_subjects = result.filter_by_topic(topic_idx=2, threshold=0.5)
print(f"Found {len(topic_2_subjects)} subjects with topic 2 weight > 0.5")
```

**Use case:** Stratify cohorts for downstream analysis (e.g., Topic-GWAS)

---

## Prediction Methods (MFVI Only)

### `predict_new_subjects(W_new, return_z=False)`
Infer topic distributions for new subjects given their disease data.

**Parameters:**
- `W_new`: Binary disease matrix for new subjects `(M_new, D)`
- `return_z`: If True, also return topic assignments

**Returns:** 
- `theta_new`: Topic distributions `(M_new, K+1)`
- `z_new` (optional): Topic assignments `(M_new, D, K+1)`

**Example:**
```python
# New patient data
W_new = np.array([[0, 1, 0, 1, 0]])  # One new patient

# Predict their topic distribution
theta_new = result.predict_new_subjects(W_new)
print(f"New patient's topics: {theta_new}")
```

**Use case:** Apply trained model to new patient data; online prediction

**Note:** This wraps the existing `infer_theta_single_row()` function

---

## Visualization Methods

### `plot_topic_heatmap(figsize=(10, 8), cmap='viridis', annotate=False)`
Visualize the beta matrix as a heatmap.

**Returns:** matplotlib Figure and Axes

**Example:**
```python
fig, ax = result.plot_topic_heatmap(annotate=True)
plt.show()
```

**Use case:** Quick visualization of topic-disease associations

---

### `plot_subject_topics(subject_ids=None, figsize=(10, 6))`
Bar plot of topic distributions for selected subjects.

**Example:**
```python
result.plot_subject_topics(subject_ids=['P001', 'P002', 'P003'])
plt.show()
```

**Use case:** Compare topic profiles across patients

---

### `plot_elbo_convergence(figsize=(10, 6))`
Plot ELBO over iterations (MFVI only).

**Example:**
```python
result.plot_elbo_convergence()
plt.show()
```

**Use case:** Diagnose convergence issues

---

### `plot_topic_distribution(figsize=(10, 6))`
Histogram showing distribution of topic prevalences in the population.

**Example:**
```python
result.plot_topic_distribution()
plt.show()
```

**Use case:** Understand how topics are distributed across subjects

---

### `plot_disease_network(topic_idx, threshold=0.5, layout='spring')`
Network graph of diseases in a topic.

**Example:**
```python
result.plot_disease_network(topic_idx=2, threshold=0.7)
plt.show()
```

**Use case:** Visualize disease relationships within a topic

---

## Serialization & Export Methods

### `to_dict(include_raw=False)`
Convert result to dictionary for serialization.

**Parameters:**
- `include_raw`: If True, include raw convergence history and samples

**Returns:** Dictionary with all result data

**Example:**
```python
data = result.to_dict()
```

---

### `save(filepath, format='pickle')`
Save result to file.

**Parameters:**
- `filepath`: Path to save file
- `format`: 'pickle', 'json', or 'hdf5'

**Example:**
```python
result.save('lfa_model.pkl')
```

---

### `load(filepath)` (classmethod)
Load result from file.

**Example:**
```python
result = LFAResult.load('lfa_model.pkl')
```

---

### `export_beta_csv(filepath)`
Export beta matrix to CSV.

**Example:**
```python
result.export_beta_csv('topic_disease_probs.csv')
```

---

### `export_theta_csv(filepath)`
Export theta matrix to CSV.

**Example:**
```python
result.export_theta_csv('subject_topic_weights.csv')
```

---

## Advanced Analysis Methods

### `compute_topic_coherence(measure='umass', W_reference=None)`
Compute topic coherence score (measure of topic quality).

**Parameters:**
- `measure`: 'umass', 'c_v', or 'c_npmi'
- `W_reference`: Reference corpus (defaults to training W)

**Returns:** Float coherence score

**Use case:** Evaluate topic quality; compare different K values

---

### `identify_rare_topic_subjects(threshold=0.1)`
Find subjects whose topic distributions are unusual/rare.

**Returns:** List of subject IDs with outlier scores

**Use case:** Find interesting/anomalous patients

---

### `get_topic_correlations()`
Compute pairwise correlations between topics.

**Returns:** Correlation matrix `(K+1, K+1)`

**Example:**
```python
corr = result.get_topic_correlations()
# High correlation suggests topics might be redundant
```

**Use case:** Assess if using too many topics (K too large)

---

### `stratify_by_topic(primary_only=True, threshold=0.3)`
Stratify subjects into topic-based groups.

**Parameters:**
- `primary_only`: If True, assign each subject to single dominant topic
- `threshold`: Minimum weight for topic assignment

**Returns:** Dictionary mapping topic_idx to list of subject_ids

**Example:**
```python
strata = result.stratify_by_topic()
for topic, subjects in strata.items():
    print(f"Topic {topic}: {len(subjects)} subjects")
```

**Use case:** Create cohorts for Topic-GWAS or other downstream analyses

---

## Implementation Notes

### Phase 4 Priority Order
1. **Tier 1 (Implement first):** 
   - `get_top_diseases_per_topic()` - Essential for interpretation
   - `get_subject_topic_distribution()` - Basic access
   - `get_disease_topic_loadings()` - Basic access
   - `summary()` - High-level overview

2. **Tier 2 (Next):**
   - `get_most_representative_subjects()`
   - `get_topic_diversity()`
   - `predict_new_subjects()` (MFVI only)
   - `to_dict()`, `save()`, `load()`

3. **Tier 3 (Later):**
   - Visualization methods
   - Advanced analysis methods
   - Export utilities

### Design Considerations
- **Consistent naming:** Use `get_` for queries, `plot_` for viz, `compute_` for expensive operations
- **Flexible identifiers:** All methods should accept both IDs (strings) and indices (ints)
- **Sensible defaults:** Methods should work with minimal parameters
- **Rich returns:** Return named tuples or dicts, not just arrays
- **Validation:** Check if disease_names/subject_ids are available before using them

### Dependencies
- Core methods: NumPy, SciPy only
- Visualization: matplotlib, seaborn (optional)
- Export: pandas (optional for CSV export)
- Advanced: networkx (optional for network plots)

Keep optional dependencies truly optional - raise informative errors if not installed.
