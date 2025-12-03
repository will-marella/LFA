# Data Format Guide

This guide explains how to prepare your data for use with the LFA package.

## Input Data Requirements

### The W Matrix (Disease Matrix)

The primary input to LFA is a **binary disease matrix** called `W`:

- **Shape**: `(M, D)` where:
  - `M` = number of subjects (patients, individuals)
  - `D` = number of diseases (conditions, diagnoses)

- **Values**: Binary (0 or 1)
  - `0` = disease absent for this subject
  - `1` = disease present for this subject

- **Format**: NumPy array or pandas DataFrame

---

## Input Format Options

### Option 1: NumPy Array (Simple)

Use a NumPy array if you don't need labeled data:

```python
import numpy as np
from lfa import fit_lfa

# Create binary disease matrix
W = np.array([
    [0, 1, 0, 1, 0],  # Subject 0: has diseases 1 and 3
    [0, 0, 1, 0, 1],  # Subject 1: has diseases 2 and 4
    [1, 1, 0, 0, 0],  # Subject 2: has diseases 0 and 1
    [0, 1, 1, 1, 0],  # Subject 3: has diseases 1, 2, and 3
])

# Fit model (uses integer indices)
result = fit_lfa(W, num_topics=2)
```

### Option 2: NumPy Array with Labels

Provide subject IDs and disease names separately:

```python
import numpy as np
from lfa import fit_lfa

# Disease data
W = np.array([
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 1],
    [1, 1, 0, 0, 0],
    [0, 1, 1, 1, 0],
])

# Labels
subject_ids = ['P001', 'P002', 'P003', 'P004']
disease_names = ['Diabetes', 'Hypertension', 'Asthma', 'CAD', 'COPD']

# Fit model with labels
result = fit_lfa(
    W, 
    num_topics=2,
    subject_ids=subject_ids,
    disease_names=disease_names
)

# Now results use your labels
print(result.disease_names)  # ['Diabetes', 'Hypertension', ...]
```

### Option 3: Pandas DataFrame (Recommended)

Use a DataFrame for automatic label extraction:

```python
import pandas as pd
from lfa import fit_lfa

# Create DataFrame with labeled rows and columns
W_df = pd.DataFrame(
    data=[
        [0, 1, 0, 1, 0],
        [0, 0, 1, 0, 1],
        [1, 1, 0, 0, 0],
        [0, 1, 1, 1, 0],
    ],
    index=['P001', 'P002', 'P003', 'P004'],  # Subject IDs
    columns=['Diabetes', 'Hypertension', 'Asthma', 'CAD', 'COPD']  # Disease names
)

# Fit model (labels extracted automatically)
result = fit_lfa(W_df, num_topics=2)

# Labels are preserved in the result
print(result.subject_ids)    # ['P001', 'P002', 'P003', 'P004']
print(result.disease_names)  # ['Diabetes', 'Hypertension', 'Asthma', 'CAD', 'COPD']
```

---

## Loading Data from Files

### From CSV

```python
import pandas as pd
from lfa import fit_lfa

# Load CSV with disease data
# Expected format: rows = subjects, columns = diseases, values = 0/1
W_df = pd.read_csv('disease_data.csv', index_col=0)

# Verify binary data
assert W_df.isin([0, 1]).all().all(), "Data must be binary (0/1)"

# Fit model
result = fit_lfa(W_df, num_topics=5)
```

Example CSV format:
```csv
Subject_ID,Diabetes,Hypertension,Asthma,CAD,COPD
P001,0,1,0,1,0
P002,0,0,1,0,1
P003,1,1,0,0,0
P004,0,1,1,1,0
```

### From Excel

```python
import pandas as pd
from lfa import fit_lfa

# Load Excel file
W_df = pd.read_excel('disease_data.xlsx', index_col=0)

# Ensure binary
W_df = W_df.astype(int)

result = fit_lfa(W_df, num_topics=5)
```

### From NumPy Binary File

```python
import numpy as np
from lfa import fit_lfa

# Load saved NumPy array
W = np.load('disease_matrix.npy')

# Load corresponding labels
subject_ids = np.load('subject_ids.npy')
disease_names = np.load('disease_names.npy')

result = fit_lfa(
    W, 
    num_topics=5,
    subject_ids=subject_ids,
    disease_names=disease_names
)
```

---

## Data Validation

The LFA package validates your input data:

### Automatic Checks

1. **Binary values**: W must contain only 0 and 1
   ```python
   # This will raise ValueError
   W_bad = np.array([[0, 1, 2], [1, 0, 1]])  # Contains 2!
   result = fit_lfa(W_bad, num_topics=2)  # Error!
   ```

2. **Shape consistency**: Subject IDs and disease names must match W dimensions
   ```python
   W = np.zeros((100, 50))  # 100 subjects, 50 diseases
   subject_ids = ['P1', 'P2']  # Wrong! Need 100 IDs
   result = fit_lfa(W, num_topics=5, subject_ids=subject_ids)  # Error!
   ```

3. **No missing values**: NaN not currently supported
   ```python
   import pandas as pd
   W_df = pd.DataFrame([[0, 1, np.nan], [1, 0, 1]])  # Has NaN
   result = fit_lfa(W_df, num_topics=2)  # Error!
   ```

### Manual Validation

```python
import numpy as np

def validate_disease_matrix(W):
    """Validate disease matrix before fitting."""
    
    # Check type
    if not isinstance(W, (np.ndarray, pd.DataFrame)):
        raise TypeError("W must be numpy array or pandas DataFrame")
    
    # Convert DataFrame to array for validation
    if isinstance(W, pd.DataFrame):
        W_array = W.values
    else:
        W_array = W
    
    # Check dimensions
    if W_array.ndim != 2:
        raise ValueError(f"W must be 2D, got shape {W_array.shape}")
    
    # Check for missing values
    if np.isnan(W_array).any():
        raise ValueError("W contains missing values (NaN)")
    
    # Check binary
    if not np.all(np.isin(W_array, [0, 1])):
        raise ValueError("W must be binary (only 0 and 1)")
    
    # Check sufficient size
    M, D = W_array.shape
    if M < 10:
        print(f"Warning: Only {M} subjects. LFA works best with M > 50")
    if D < 5:
        print(f"Warning: Only {D} diseases. LFA works best with D > 10")
    
    return True

# Use before fitting
validate_disease_matrix(W)
result = fit_lfa(W, num_topics=5)
```

---

## Data Preparation Tips

### 1. Handling Missing Data

LFA currently does not handle missing data. You must impute before fitting:

```python
import pandas as pd
import numpy as np

# Load data with missing values
W_df = pd.read_csv('data_with_missing.csv', index_col=0)

# Option A: Remove rows with any missing values
W_clean = W_df.dropna()

# Option B: Remove columns with too many missing values
W_clean = W_df.dropna(axis=1, thresh=int(0.9 * len(W_df)))

# Option C: Impute with most common value (0 for diseases)
W_clean = W_df.fillna(0).astype(int)

# Option D: Impute based on prevalence
for col in W_df.columns:
    prevalence = W_df[col].mean()
    W_df[col].fillna(int(prevalence > 0.5), inplace=True)

# Now fit
result = fit_lfa(W_clean, num_topics=5)
```

### 2. Filtering Rare/Common Diseases

Remove diseases that are too rare or too common:

```python
import pandas as pd

W_df = pd.read_csv('disease_data.csv', index_col=0)

# Calculate disease prevalence
prevalence = W_df.mean()

# Keep diseases with 5% < prevalence < 95%
diseases_to_keep = prevalence[(prevalence > 0.05) & (prevalence < 0.95)]
W_filtered = W_df[diseases_to_keep.index]

print(f"Kept {len(W_filtered.columns)} of {len(W_df.columns)} diseases")

result = fit_lfa(W_filtered, num_topics=5)
```

### 3. Handling Sparse Data

For very sparse disease matrices:

```python
import numpy as np
from scipy.sparse import csr_matrix

# If your data is very sparse, you might start with sparse format
# But LFA requires dense arrays, so convert:
W_sparse = csr_matrix(...)  # Your sparse matrix
W_dense = W_sparse.toarray()

# Check sparsity
sparsity = 1 - (np.count_nonzero(W_dense) / W_dense.size)
print(f"Data is {sparsity*100:.1f}% sparse")

if sparsity > 0.95:
    print("Warning: Very sparse data. Consider filtering or using fewer topics.")

result = fit_lfa(W_dense, num_topics=5)
```

### 4. Aggregating Diagnoses

If you have hierarchical disease codes (e.g., ICD codes), you may want to aggregate:

```python
import pandas as pd

# Example: Aggregate ICD-10 codes to broader categories
W_detailed = pd.read_csv('icd10_data.csv', index_col=0)

# Mapping from detailed to broad categories
disease_mapping = {
    'E11.0': 'Diabetes',
    'E11.1': 'Diabetes',
    'E11.2': 'Diabetes',
    'I10': 'Hypertension',
    'I11': 'Hypertension',
    'J45.0': 'Asthma',
    'J45.1': 'Asthma',
    # ... more mappings
}

# Aggregate
W_aggregated = pd.DataFrame()
for broad_category in set(disease_mapping.values()):
    detailed_codes = [k for k, v in disease_mapping.items() if v == broad_category]
    # Subject has broad category if they have ANY detailed code
    W_aggregated[broad_category] = W_detailed[detailed_codes].max(axis=1)

result = fit_lfa(W_aggregated, num_topics=5)
```

---

## Recommended Data Sizes

For best results:

| Algorithm | Min Subjects | Recommended Subjects | Min Diseases | Recommended Diseases |
|-----------|-------------|---------------------|--------------|---------------------|
| MFVI      | 50          | 500+                | 10           | 20+                 |
| PCGS      | 20          | 100-500             | 10           | 20+                 |

**Notes:**
- MFVI scales well to large datasets (M > 10,000)
- PCGS is slower but more accurate for small datasets (M < 500)
- Number of topics K should be much smaller than D (typically K < D/2)
- More diseases (D) generally improves topic interpretability

---

## Example Workflows

### Workflow 1: Simple CSV to Results

```python
import pandas as pd
from lfa import fit_lfa

# Load data
W = pd.read_csv('my_disease_data.csv', index_col=0)

# Quick validation
assert W.isin([0, 1]).all().all()
print(f"Data shape: {W.shape}")

# Fit
result = fit_lfa(W, num_topics=8, algorithm='mfvi')

# Analyze
print(result.summary())
```

### Workflow 2: With Preprocessing

```python
import pandas as pd
from lfa import fit_lfa, select_num_topics

# Load
W = pd.read_csv('raw_disease_data.csv', index_col=0)

# Preprocess
W = W.dropna()  # Remove missing
prevalence = W.mean()
W = W.loc[:, (prevalence > 0.05) & (prevalence < 0.95)]  # Filter diseases
print(f"After filtering: {W.shape}")

# Auto-select K
best_k, cv_results = select_num_topics(W, candidate_topics=[5, 8, 10, 15])
print(f"Best K: {best_k}")

# Fit final model
result = fit_lfa(W, num_topics=best_k, algorithm='mfvi')
result.summary()
```

---

## See Also

- [API Documentation](API.md) - Function signatures and parameters
- [Potential Output Methods](POTENTIAL_OUTPUT_METHODS.md) - Analyzing results
