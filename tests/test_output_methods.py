"""Tests for LFAResult output methods (Tier 1)."""

import pytest
import numpy as np

from lfa import fit_lfa, simulate_topic_disease_data


def test_get_top_diseases_per_topic_basic():
    """Test get_top_diseases_per_topic with default parameters."""
    W, beta, theta, z = simulate_topic_disease_data(seed=42, M=40, D=12, K=2)
    result = fit_lfa(W, num_topics=2, algorithm='mfvi', max_iterations=50, verbose=False)
    
    top_diseases = result.get_top_diseases_per_topic(n=5)
    
    # Should return dict with 2 topics (excluding healthy topic by default)
    assert len(top_diseases) == 2
    assert 1 in top_diseases
    assert 2 in top_diseases
    
    # Each topic should have 5 diseases
    for topic_idx, diseases in top_diseases.items():
        assert len(diseases) == 5
        # Each entry should be (disease, probability) tuple
        for disease, prob in diseases:
            assert isinstance(disease, int)  # Default uses indices
            assert isinstance(prob, (float, np.floating))
            assert 0 <= prob <= 1
        
        # Probabilities should be sorted descending
        probs = [p for _, p in diseases]
        assert probs == sorted(probs, reverse=True)


def test_get_top_diseases_per_topic_with_names():
    """Test get_top_diseases_per_topic with disease names."""
    W, beta, theta, z = simulate_topic_disease_data(seed=42, M=40, D=8, K=2)
    
    disease_names = ['D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8']
    result = fit_lfa(W, num_topics=2, disease_names=disease_names, 
                     algorithm='mfvi', max_iterations=50, verbose=False)
    
    top_diseases = result.get_top_diseases_per_topic(n=3, use_names=True)
    
    # Should have disease names, not indices
    for topic_idx, diseases in top_diseases.items():
        for disease, prob in diseases:
            assert isinstance(disease, str)
            assert disease in disease_names


def test_get_top_diseases_per_topic_include_healthy():
    """Test get_top_diseases_per_topic including healthy topic."""
    W, beta, theta, z = simulate_topic_disease_data(seed=42, M=40, D=10, K=2)
    result = fit_lfa(W, num_topics=2, algorithm='mfvi', max_iterations=50, verbose=False)
    
    # Include healthy topic
    top_diseases = result.get_top_diseases_per_topic(n=4, exclude_healthy=False)
    
    # Should have 3 topics (healthy + 2 disease topics)
    assert len(top_diseases) == 3
    assert 0 in top_diseases  # Healthy topic
    assert 1 in top_diseases
    assert 2 in top_diseases


def test_get_top_diseases_per_topic_edge_cases():
    """Test edge cases for get_top_diseases_per_topic."""
    W, beta, theta, z = simulate_topic_disease_data(seed=42, M=30, D=5, K=1)
    result = fit_lfa(W, num_topics=1, algorithm='mfvi', max_iterations=50, verbose=False)
    
    # Request more diseases than available
    top_diseases = result.get_top_diseases_per_topic(n=10)
    
    # Should return all 5 diseases (capped at D)
    for topic_idx, diseases in top_diseases.items():
        assert len(diseases) == 5  # All diseases returned


def test_get_disease_topic_loadings_by_index():
    """Test get_disease_topic_loadings using disease_idx."""
    W, beta, theta, z = simulate_topic_disease_data(seed=42, M=40, D=10, K=2)
    result = fit_lfa(W, num_topics=2, algorithm='mfvi', max_iterations=50, verbose=False)
    
    # Get loadings for disease 0
    loadings = result.get_disease_topic_loadings(disease_idx=0)
    
    # Should return array of shape (K+1,) = (3,)
    assert loadings.shape == (3,)
    
    # Values should match beta[:, 0]
    np.testing.assert_array_equal(loadings, result.beta[:, 0])
    
    # Should be a copy, not a view
    assert loadings is not result.beta[:, 0]


def test_get_disease_topic_loadings_by_name():
    """Test get_disease_topic_loadings using disease_name."""
    W, beta, theta, z = simulate_topic_disease_data(seed=42, M=40, D=6, K=2)
    
    disease_names = ['Diabetes', 'Hypertension', 'Obesity', 'Asthma', 'Depression', 'Arthritis']
    result = fit_lfa(W, num_topics=2, disease_names=disease_names,
                     algorithm='mfvi', max_iterations=50, verbose=False)
    
    # Get loadings by name
    loadings = result.get_disease_topic_loadings(disease_name='Diabetes')
    
    # Should return array of shape (3,)
    assert loadings.shape == (3,)
    
    # Should match loadings by index
    loadings_by_idx = result.get_disease_topic_loadings(disease_idx=0)
    np.testing.assert_array_equal(loadings, loadings_by_idx)


def test_get_disease_topic_loadings_integer_names():
    """Test get_disease_topic_loadings with integer disease names."""
    W, beta, theta, z = simulate_topic_disease_data(seed=42, M=40, D=8, K=2)
    result = fit_lfa(W, num_topics=2, algorithm='mfvi', max_iterations=50, verbose=False)
    
    # Default disease_names are integers [0, 1, 2, ...]
    # Should be able to look up by integer name
    loadings = result.get_disease_topic_loadings(disease_name=0)
    
    assert loadings.shape == (3,)
    np.testing.assert_array_equal(loadings, result.beta[:, 0])


def test_get_disease_topic_loadings_error_handling():
    """Test error handling in get_disease_topic_loadings."""
    W, beta, theta, z = simulate_topic_disease_data(seed=42, M=40, D=10, K=2)
    
    disease_names = [f'Disease_{i}' for i in range(10)]
    result = fit_lfa(W, num_topics=2, disease_names=disease_names,
                     algorithm='mfvi', max_iterations=50, verbose=False)
    
    # Error: neither parameter provided
    with pytest.raises(ValueError, match="Must provide either"):
        result.get_disease_topic_loadings()
    
    # Error: both parameters provided
    with pytest.raises(ValueError, match="Cannot provide both"):
        result.get_disease_topic_loadings(disease_name='Disease_0', disease_idx=0)
    
    # Error: disease not found
    with pytest.raises(ValueError, match="not found"):
        result.get_disease_topic_loadings(disease_name='NotADisease')
    
    # Error: index out of range (negative)
    with pytest.raises(ValueError, match="out of range"):
        result.get_disease_topic_loadings(disease_idx=-1)
    
    # Error: index out of range (too large)
    with pytest.raises(ValueError, match="out of range"):
        result.get_disease_topic_loadings(disease_idx=100)


def test_output_methods_with_pcgs():
    """Test that output methods work with PCGS results."""
    W, beta, theta, z = simulate_topic_disease_data(seed=42, M=30, D=10, K=2)
    
    result = fit_lfa(W, num_topics=2, algorithm='pcgs', 
                     num_chains=2, max_iterations=100, 
                     post_convergence_samples=10, verbose=False)
    
    # Both methods should work with PCGS results
    top_diseases = result.get_top_diseases_per_topic(n=5)
    assert len(top_diseases) == 2
    
    loadings = result.get_disease_topic_loadings(disease_idx=0)
    assert loadings.shape == (3,)


def test_output_methods_integration():
    """Integration test: use output methods together."""
    W, beta, theta, z = simulate_topic_disease_data(seed=42, M=40, D=12, K=3)
    
    disease_names = [f'Disease_{chr(65+i)}' for i in range(12)]  # Disease_A, Disease_B, ...
    result = fit_lfa(W, num_topics=3, disease_names=disease_names,
                     algorithm='mfvi', max_iterations=100, verbose=False)
    
    # Get top diseases
    top_diseases = result.get_top_diseases_per_topic(n=3)
    
    # For each topic, check the top disease loadings
    for topic_idx, diseases in top_diseases.items():
        top_disease_name, top_prob = diseases[0]  # Highest probability disease
        
        # Get full loadings for that disease
        loadings = result.get_disease_topic_loadings(disease_name=top_disease_name)
        
        # The topic_idx should have high loading for this disease
        # (accounting for 0-indexing: topic 1 = index 1, topic 2 = index 2, etc.)
        assert loadings[topic_idx] == top_prob


if __name__ == '__main__':
    # Run tests
    print("Running Tier 1 output method tests...\n")
    
    test_get_top_diseases_per_topic_basic()
    print("✓ test_get_top_diseases_per_topic_basic")
    
    test_get_top_diseases_per_topic_with_names()
    print("✓ test_get_top_diseases_per_topic_with_names")
    
    test_get_top_diseases_per_topic_include_healthy()
    print("✓ test_get_top_diseases_per_topic_include_healthy")
    
    test_get_top_diseases_per_topic_edge_cases()
    print("✓ test_get_top_diseases_per_topic_edge_cases")
    
    test_get_disease_topic_loadings_by_index()
    print("✓ test_get_disease_topic_loadings_by_index")
    
    test_get_disease_topic_loadings_by_name()
    print("✓ test_get_disease_topic_loadings_by_name")
    
    test_get_disease_topic_loadings_integer_names()
    print("✓ test_get_disease_topic_loadings_integer_names")
    
    test_get_disease_topic_loadings_error_handling()
    print("✓ test_get_disease_topic_loadings_error_handling")
    
    test_output_methods_with_pcgs()
    print("✓ test_output_methods_with_pcgs")
    
    test_output_methods_integration()
    print("✓ test_output_methods_integration")
    
    print("\n✅ All output method tests passed!")
