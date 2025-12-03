# LFA Refactoring Log

This document tracks the refactoring process to create a user-friendly API for the LFA package.

## Goals

1. **Clean User API**: Create `fit_lfa()` and `select_num_topics()` that only require data (W) and number of topics (K)
2. **LFAResult Class**: Encapsulate results with rich analysis methods
3. **Separate Inference from Evaluation**: Ground truth parameters only needed for research/experiments
4. **Better I/O**: Support DataFrames, preserve subject/disease identifiers
5. **Documentation**: Clear guides for users vs. researchers

## Design Decisions

### 1. Input Format Support
**Decision:** Support both NumPy arrays and pandas DataFrames
- NumPy arrays: Simple, no dependencies
- DataFrames: Automatic label extraction
- Optional `subject_ids` and `disease_names` parameters for NumPy arrays

**Rationale:** Flexibility without forcing pandas dependency; researchers often have labeled data

### 2. Research vs. User APIs
**Decision:** Separate modules (Option B from planning)
- User API: `fit_lfa()` in `src/lfa.py` - no ground truth needed
- Research API: Functions in `src/evaluation/` for comparing to ground truth
- Evaluation as separate step after fitting

**Rationale:** Cleanest separation; most flexible; doesn't confuse end users

### 3. Healthy Topic Handling
**Decision:** Always include healthy topic internally (K+1), make it transparent to users
- User specifies K (number of disease topics)
- Model internally uses K+1 (includes healthy topic)
- Results clearly label which topic is "healthy"

**Rationale:** Simplifies user interface; healthy topic is implementation detail for disease modeling

### 4. Default Hyperparameters
**Decision:** 
- `alpha`: Default to 0.1 (uniform prior)
- `max_iterations`: 1000 for MFVI, 3000 for PCGS
- `convergence_threshold`: 1e-6 for MFVI
- `num_chains`: 3 for PCGS

**Rationale:** Based on existing experiment scripts; proven to work well

### 5. PCGS Multi-Chain Results
**Decision:** Store both per-chain and merged results in `LFAResult`
- Primary attributes (`beta`, `theta`, `z`) contain merged/averaged results
- Raw chain samples available in `result.raw_samples` dict (if PCGS)
- Leverage existing chain alignment and merging logic

**Rationale:** Provides both convenience (merged) and flexibility (raw chains) for researchers

### 6. Auto-K Selection
**Decision:** Only support MFVI for auto-K selection
- PCGS too slow for cross-validation
- Current BIC implementation flagged as needing validation (future work)
- Default candidate_topics: [3, 5, 8, 10, 15, 20]

**Rationale:** Practical computational constraints; MFVI is fast enough for CV

## Project Structure Changes

### New Files Created
```
LFA/
├── src/
│   ├── lfa.py                    # NEW: Main user API (fit_lfa, select_num_topics)
│   ├── core/
│   │   ├── __init__.py           # NEW
│   │   ├── results.py            # NEW: LFAResult class
│   │   └── inference.py          # NEW: Clean inference wrappers
│   └── evaluation/               # NEW: Research/evaluation functions
│       ├── __init__.py
│       └── metrics.py            # Evaluation with ground truth
├── docs/                         # NEW: Documentation
│   ├── API.md
│   ├── DATA_FORMAT.md
│   ├── POTENTIAL_OUTPUT_METHODS.md
│   └── REFACTORING.md (this file)
└── examples/                     # NEW: User-facing examples (future)
```

### Deprecated/Moved Files
- `src/mfvi_sampler.py` - **Wrapped** by `src/core/inference.py`, kept for backward compat
- `src/gibbs_sampler.py` - **Wrapped** by `src/core/inference.py`, kept for backward compat
- `src/scripts/*` - **Kept as-is** for now, may move to `src/experiments/` later

### Unchanged Files (Core Logic)
**These files contain working MFVI/PCGS logic and MUST NOT be modified:**
- `src/models/mfvi_model.py` - Core MFVI implementation
- `src/models/pcgs.py` - Core PCGS implementation
- `src/utils/mfvi_monitor.py` - ELBO monitoring
- `src/utils/pcgs_monitor.py` - R-hat monitoring
- `src/experiment/simulation.py` - Data generation

**Rationale:** These implement validated algorithms; refactoring is about API, not algorithms

## Implementation Phases

### Phase 1: Foundation (Current Priority)
- [x] Create `docs/` with API specifications
- [ ] Create `src/core/results.py` with `LFAResult` class (minimal version)
- [ ] Create `src/core/inference.py` with clean inference functions
- [ ] Create `src/lfa.py` with `fit_lfa()` and `select_num_topics()`
- [ ] Test new API with synthetic data

**Testing approach:** Use existing `simulate_topic_disease_data()` for tests; numerical results should match old API

### Phase 2: Evaluation Separation
- [ ] Create `src/evaluation/metrics.py` for ground truth evaluation
- [ ] Update experiment scripts to use new evaluation module
- [ ] Document research API usage

### Phase 3: Enhanced Output Methods
- [ ] Implement Tier 1 methods in `LFAResult`:
  - `get_top_diseases_per_topic()`
  - `get_subject_topic_distribution()`
  - `get_disease_topic_loadings()`
  - `summary()`
- [ ] Implement serialization: `to_dict()`, `save()`, `load()`

### Phase 4: Examples and Documentation
- [ ] Create `examples/` directory with user workflows
- [ ] Update main README.md with new API
- [ ] Add notebook examples (optional)

### Phase 5: Optional Enhancements
- [ ] Prediction for new subjects (MFVI only)
- [ ] Visualization methods
- [ ] Advanced analysis methods

## Backward Compatibility

### Strategy
- **Keep old functions:** `src/mfvi_sampler.py` and `src/gibbs_sampler.py` remain unchanged
- **New API wraps old:** `src/core/inference.py` wraps existing functions
- **Experiment scripts:** Can be updated or kept as-is; they're internal tools
- **Tests:** Will update to use new API, but can test equivalence with old API

### Breaking Changes (Acceptable)
- Experiment scripts may need import path updates
- Old test files may need updates
- Internal utilities may be reorganized

### Non-Breaking (Maintained)
- Core algorithm implementations stay identical
- Numerical outputs stay identical
- Data generation functions stay identical

## Testing Strategy

### Unit Tests
- `LFAResult` class methods
- Input validation in `fit_lfa()`
- DataFrame vs. NumPy array handling

### Integration Tests
- Full pipeline: W → fit_lfa() → LFAResult → analysis methods
- Numerical equivalence: new API vs. old API results
- MFVI and PCGS both tested

### Test Data
- Use existing `simulate_topic_disease_data()`
- Small matrices (M=50, D=20, K=3) for fast tests
- Fixed seeds for reproducibility

### Acceptance Criteria
- New API produces numerically identical beta, theta, z as old API
- All core algorithm logic unchanged
- DataFrames preserve and use labels correctly
- No regression in existing functionality

## Key Constraints

### Must Not Change
1. **MFVI algorithm logic** (src/models/mfvi_model.py)
2. **PCGS algorithm logic** (src/models/pcgs.py)
3. **Convergence monitoring** (src/utils/*_monitor.py)
4. **Data simulation** (src/experiment/simulation.py)

### Must Preserve
1. **Numerical accuracy** - Results identical to old API
2. **Convergence behavior** - Same stopping criteria
3. **Chain alignment** - Existing PCGS chain merging logic

### Can Change
1. **API signatures** - Making them user-friendly
2. **File organization** - Better structure
3. **Documentation** - Much improved
4. **I/O handling** - Support DataFrames, labels
5. **Error messages** - More informative

## Open Questions / Future Work

### 1. Auto-K Selection
- **Issue:** Current BIC implementation may not select correct K
- **Status:** Flagged in documentation; will revisit later
- **Workaround:** Users can manually try different K values; provide guidance on reasonable ranges

### 2. Healthy Topic
- **Question:** Should it be optional or always included?
- **Current:** Always included (K+1), transparent to users
- **Future:** Could add `include_healthy_topic` parameter if needed

### 3. Missing Data
- **Status:** Not currently supported
- **Future:** Could add imputation or allow NaN in W with special handling

### 4. Sparse Data
- **Status:** Requires dense arrays currently
- **Future:** Could optimize for sparse matrices (large computational change)

### 5. Online/Streaming Updates
- **Status:** Batch processing only
- **Future:** Could add incremental updates for new subjects (MFVI only)

## Timeline & Milestones

### Milestone 1: Minimal Viable API ✓ (In Progress)
- Core documentation complete
- LFAResult class implemented
- fit_lfa() working
- Basic tests passing

### Milestone 2: Feature Complete
- All Tier 1 output methods implemented
- Evaluation module functional
- Examples created
- README updated

### Milestone 3: Polish
- All documentation complete
- Edge cases handled
- Error messages improved
- Performance optimized

## Notes for Future Maintainers

1. **Don't touch core algorithms:** The MFVI and PCGS implementations are validated. Any changes risk breaking correctness.

2. **Test numerical equivalence:** When refactoring, always compare outputs to original implementation with fixed seeds.

3. **Preserve chain alignment:** PCGS chain merging is subtle and works well; don't "simplify" it without careful validation.

4. **Document breaking changes:** If API changes, update AGENTS.md and this file.

5. **Keep auto-K experimental:** Until BIC validation is done, don't promise robust K selection.

## References

- Original codebase: `/Users/willcambridge/CamResearch/Main/LFA/src/`
- Experiment scripts: `/Users/willcambridge/CamResearch/Main/experiments/`
- Test examples: `/Users/willcambridge/CamResearch/Main/LFA/tests/`

## Change Log

### 2024-12-03: Initial Refactoring Plan
- Created documentation structure
- Defined API specifications
- Outlined implementation phases
- Identified constraints and non-negotiables
