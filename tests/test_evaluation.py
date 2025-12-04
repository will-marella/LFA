"""Tests for the evaluation API."""

import pytest
import numpy as np

from lfa import fit_lfa, simulate_topic_disease_data
from lfa.evaluation import evaluate_result, compare_algorithms


def test_evaluate_mfvi_basic():
    """Test MFVI evaluation produces expected metrics."""
    # Simulate small dataset
    W, true_beta, true_theta, z = simulate_topic_disease_data(
        seed=42, M=50, D=12, K=2
    )
    
    # Fit MFVI with limited iterations for speed
    result = fit_lfa(
        W, num_topics=2, algorithm='mfvi',
        max_iterations=100, verbose=False
    )
    
    # Evaluate
    metrics = evaluate_result(result, true_beta, true_theta)
    
    # Check all expected metrics are present
    assert 'beta_correlation' in metrics
    assert 'theta_correlation' in metrics
    assert 'beta_mse' in metrics
    assert 'theta_mse' in metrics
    assert 'num_iterations' in metrics
    assert 'final_elbo' in metrics
    assert 'converged' in metrics
    
    # Check metrics are reasonable
    assert isinstance(metrics['beta_correlation'], float)
    assert isinstance(metrics['theta_correlation'], float)
    assert metrics['beta_mse'] >= 0
    assert metrics['theta_mse'] >= 0
    assert metrics['num_iterations'] > 0
    assert metrics['converged'] in [True, False]


def test_evaluate_pcgs_basic():
    """Test PCGS evaluation produces expected metrics."""
    # Simulate small dataset
    W, true_beta, true_theta, z = simulate_topic_disease_data(
        seed=43, M=50, D=12, K=2
    )
    
    # Fit PCGS with minimal settings for speed
    result = fit_lfa(
        W, num_topics=2, algorithm='pcgs',
        num_chains=2, max_iterations=100,
        post_convergence_samples=10,
        verbose=False
    )
    
    # Evaluate
    metrics = evaluate_result(result, true_beta, true_theta)
    
    # Check all expected metrics are present
    assert 'beta_mae' in metrics
    assert 'beta_pearson_corr' in metrics
    assert 'theta_mae' in metrics
    assert 'theta_pearson_corr' in metrics
    assert 'num_iterations' in metrics
    assert 'r_hat_beta' in metrics
    assert 'r_hat_theta' in metrics
    assert 'r_hat_overall' in metrics
    assert 'converged' in metrics
    
    # Check metrics are reasonable
    assert metrics['beta_mae'] >= 0
    assert metrics['theta_mae'] >= 0
    assert isinstance(metrics['beta_pearson_corr'], float)
    assert isinstance(metrics['theta_pearson_corr'], float)
    assert metrics['num_iterations'] > 0
    assert metrics['converged'] in [True, False]


def test_evaluate_does_not_modify_original():
    """Ensure evaluation doesn't modify the original result."""
    W, true_beta, true_theta, z = simulate_topic_disease_data(
        seed=44, M=30, D=10, K=2
    )
    
    result = fit_lfa(
        W, num_topics=2, algorithm='mfvi',
        max_iterations=50, verbose=False
    )
    
    # Store copies of original arrays
    beta_before = result.beta.copy()
    theta_before = result.theta.copy()
    z_before = result.z.copy()
    
    # Evaluate
    metrics = evaluate_result(result, true_beta, true_theta)
    
    # Check that result arrays are unchanged
    np.testing.assert_array_equal(result.beta, beta_before)
    np.testing.assert_array_equal(result.theta, theta_before)
    np.testing.assert_array_equal(result.z, z_before)


def test_compare_algorithms_basic():
    """Test algorithm comparison returns DataFrame."""
    W, true_beta, true_theta, z = simulate_topic_disease_data(
        seed=45, M=40, D=10, K=2
    )
    
    # Compare with minimal settings
    comparison = compare_algorithms(
        W, num_topics=2,
        true_beta=true_beta,
        true_theta=true_theta,
        algorithms=['mfvi', 'pcgs'],
        mfvi_max_iterations=50,
        mfvi_verbose=False,
        pcgs_num_chains=2,
        pcgs_max_iterations=50,
        pcgs_post_convergence_samples=5,
        pcgs_verbose=False
    )
    
    # Check it's a DataFrame with expected structure
    assert len(comparison) == 2
    assert 'algorithm' in comparison.columns
    assert 'runtime' in comparison.columns
    assert 'converged' in comparison.columns
    
    # Check both algorithms are present
    assert 'mfvi' in comparison['algorithm'].values
    assert 'pcgs' in comparison['algorithm'].values
    
    # Check runtimes are positive
    assert all(comparison['runtime'] > 0)


def test_compare_algorithms_single():
    """Test comparison with single algorithm."""
    W, true_beta, true_theta, z = simulate_topic_disease_data(
        seed=46, M=30, D=10, K=2
    )
    
    # Compare only MFVI
    comparison = compare_algorithms(
        W, num_topics=2,
        true_beta=true_beta,
        true_theta=true_theta,
        algorithms=['mfvi'],
        mfvi_max_iterations=50,
        mfvi_verbose=False
    )
    
    assert len(comparison) == 1
    assert comparison['algorithm'].iloc[0] == 'mfvi'
    assert 'beta_correlation' in comparison.columns


def test_evaluate_with_different_shapes():
    """Test evaluation handles different data dimensions."""
    # Test with K=1 (minimal topics)
    W1, beta1, theta1, z1 = simulate_topic_disease_data(
        seed=47, M=30, D=8, K=1
    )
    result1 = fit_lfa(W1, num_topics=1, algorithm='mfvi', max_iterations=50, verbose=False)
    metrics1 = evaluate_result(result1, beta1, theta1)
    assert 'beta_correlation' in metrics1
    
    # Test with K=4 (more topics)
    W4, beta4, theta4, z4 = simulate_topic_disease_data(
        seed=48, M=50, D=15, K=4
    )
    result4 = fit_lfa(W4, num_topics=4, algorithm='mfvi', max_iterations=50, verbose=False)
    metrics4 = evaluate_result(result4, beta4, theta4)
    assert 'beta_correlation' in metrics4


def test_evaluate_invalid_algorithm():
    """Test that evaluation fails gracefully with invalid algorithm."""
    W, true_beta, true_theta, z = simulate_topic_disease_data(
        seed=49, M=30, D=10, K=2
    )
    
    result = fit_lfa(W, num_topics=2, algorithm='mfvi', max_iterations=50, verbose=False)
    
    # Manually corrupt algorithm field
    result.algorithm = 'invalid_algo'
    
    with pytest.raises(ValueError, match="Unknown algorithm"):
        evaluate_result(result, true_beta, true_theta)


if __name__ == '__main__':
    # Run tests
    test_evaluate_mfvi_basic()
    print("✓ test_evaluate_mfvi_basic passed")
    
    test_evaluate_pcgs_basic()
    print("✓ test_evaluate_pcgs_basic passed")
    
    test_evaluate_does_not_modify_original()
    print("✓ test_evaluate_does_not_modify_original passed")
    
    test_compare_algorithms_basic()
    print("✓ test_compare_algorithms_basic passed")
    
    test_compare_algorithms_single()
    print("✓ test_compare_algorithms_single passed")
    
    test_evaluate_with_different_shapes()
    print("✓ test_evaluate_with_different_shapes passed")
    
    test_evaluate_invalid_algorithm()
    print("✓ test_evaluate_invalid_algorithm passed")
    
    print("\n✅ All evaluation tests passed!")
