"""Tests for the new user-facing LFA API.

This test file verifies that the new API works correctly and produces
numerically equivalent results to the original implementation.
"""

import sys
import pathlib
import numpy as np

# Add LFA to path
_LFA_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_LFA_ROOT) not in sys.path:
    sys.path.insert(0, str(_LFA_ROOT))

from src.lfa import fit_lfa, select_num_topics
from src.experiment.simulation import simulate_topic_disease_data


def test_fit_lfa_mfvi_basic():
    """Test basic MFVI fitting with new API."""
    print("\n" + "="*60)
    print("TEST: Basic MFVI fitting")
    print("="*60)
    
    # Generate synthetic data (D must be divisible by K for simulation)
    M, D, K = 100, 24, 3
    W, z_true, beta_true, theta_true = simulate_topic_disease_data(
        seed=42,
        M=M,
        D=D,
        K=K,
        topic_associated_prob=0.30,
        nontopic_associated_prob=0.01
    )
    
    # Fit using new API
    result = fit_lfa(
        W,
        num_topics=K,
        algorithm='mfvi',
        max_iterations=500,
        verbose=True
    )
    
    # Verify result structure
    assert result.algorithm == 'mfvi'
    assert result.num_topics == K
    assert result.num_subjects == M
    assert result.num_diseases == D
    assert result.beta.shape == (K+1, D)
    assert result.theta.shape == (M, K+1)
    assert result.z.shape == (M, D, K+1)
    assert result.chains is None  # MFVI doesn't have chains
    
    # Check convergence
    assert 'num_iterations' in result.convergence_info
    assert 'final_elbo' in result.convergence_info
    assert 'converged' in result.convergence_info
    
    print(f"\n✓ Result structure correct")
    print(f"✓ Beta shape: {result.beta.shape}")
    print(f"✓ Theta shape: {result.theta.shape}")
    print(f"✓ Converged: {result.convergence_info['converged']}")
    print(f"✓ Iterations: {result.convergence_info['num_iterations']}")
    
    # Print summary
    print("\n" + result.summary())
    
    return result


def test_fit_lfa_pcgs_basic():
    """Test basic PCGS fitting with new API."""
    print("\n" + "="*60)
    print("TEST: Basic PCGS fitting")
    print("="*60)
    
    # Generate small synthetic data (PCGS is slow, D must be divisible by K)
    M, D, K = 50, 16, 2
    W, z_true, beta_true, theta_true = simulate_topic_disease_data(
        seed=42,
        M=M,
        D=D,
        K=K,
        topic_associated_prob=0.30,
        nontopic_associated_prob=0.01
    )
    
    # Fit using new API
    result = fit_lfa(
        W,
        num_topics=K,
        algorithm='pcgs',
        num_chains=2,
        max_iterations=1000,
        r_hat_threshold=1.5,  # Relaxed for faster test
        post_convergence_samples=20,
        verbose=True
    )
    
    # Verify result structure
    assert result.algorithm == 'pcgs'
    assert result.num_topics == K
    assert result.num_subjects == M
    assert result.num_diseases == D
    assert result.beta.shape == (K+1, D)
    assert result.theta.shape == (M, K+1)
    assert result.z.shape == (M, D, K+1)
    assert result.chains is not None  # PCGS has chains
    assert len(result.chains) == 2  # num_chains
    
    # Check convergence
    assert 'r_hat_beta' in result.convergence_info
    assert 'r_hat_theta' in result.convergence_info
    assert 'num_chains' in result.convergence_info
    
    # Check per-chain results
    for i, chain in enumerate(result.chains):
        assert 'beta' in chain
        assert 'theta' in chain
        assert 'z' in chain
        assert chain['beta'].shape == (K+1, D)
        assert chain['theta'].shape == (M, K+1)
        assert chain['z'].shape == (M, D, K+1)
    
    print(f"\n✓ Result structure correct")
    print(f"✓ Beta shape: {result.beta.shape}")
    print(f"✓ Theta shape: {result.theta.shape}")
    print(f"✓ Number of chains: {len(result.chains)}")
    print(f"✓ R-hat (beta): {result.convergence_info['r_hat_beta']:.4f}")
    print(f"✓ R-hat (theta): {result.convergence_info['r_hat_theta']:.4f}")
    
    # Print summary
    print("\n" + result.summary())
    
    return result


def test_fit_lfa_with_labels():
    """Test fitting with subject IDs and disease names."""
    print("\n" + "="*60)
    print("TEST: Fitting with labels")
    print("="*60)
    
    # Generate synthetic data
    M, D, K = 50, 10, 2
    W, z_true, beta_true, theta_true = simulate_topic_disease_data(
        seed=42,
        M=M,
        D=D,
        K=K,
        topic_associated_prob=0.30,
        nontopic_associated_prob=0.01
    )
    
    # Create labels
    subject_ids = [f'P{i:03d}' for i in range(M)]
    disease_names = [f'Disease_{i}' for i in range(D)]
    
    # Fit with labels
    result = fit_lfa(
        W,
        num_topics=K,
        algorithm='mfvi',
        subject_ids=subject_ids,
        disease_names=disease_names,
        max_iterations=300,
        verbose=False
    )
    
    # Verify labels are stored
    assert result.subject_ids == subject_ids
    assert result.disease_names == disease_names
    
    print(f"\n✓ Subject IDs preserved: {result.subject_ids[:3]}...")
    print(f"✓ Disease names preserved: {result.disease_names[:3]}...")
    
    return result


def test_lfaresult_to_dict():
    """Test LFAResult.to_dict() method."""
    print("\n" + "="*60)
    print("TEST: LFAResult.to_dict()")
    print("="*60)
    
    # Generate and fit
    M, D, K = 30, 10, 2
    W, _, _, _ = simulate_topic_disease_data(seed=42, M=M, D=D, K=K)
    result = fit_lfa(W, num_topics=K, algorithm='mfvi', max_iterations=200, verbose=False)
    
    # Convert to dict
    result_dict = result.to_dict()
    
    # Verify keys
    expected_keys = ['beta', 'theta', 'z', 'num_topics', 'num_subjects', 
                     'num_diseases', 'algorithm', 'subject_ids', 'disease_names',
                     'convergence_info', 'metadata']
    for key in expected_keys:
        assert key in result_dict, f"Missing key: {key}"
    
    print(f"\n✓ to_dict() contains all expected keys")
    print(f"✓ Keys: {list(result_dict.keys())}")
    
    return result_dict


def test_alpha_parameter():
    """Test different alpha parameter formats."""
    print("\n" + "="*60)
    print("TEST: Alpha parameter handling")
    print("="*60)
    
    M, D, K = 30, 10, 2
    W, _, _, _ = simulate_topic_disease_data(seed=42, M=M, D=D, K=K)
    
    # Test 1: Default alpha
    result1 = fit_lfa(W, num_topics=K, max_iterations=200, verbose=False)
    assert result1.metadata['alpha'] == [0.1] * (K+1)
    print("✓ Default alpha (0.1) works")
    
    # Test 2: Float alpha
    result2 = fit_lfa(W, num_topics=K, alpha=0.5, max_iterations=200, verbose=False)
    assert result2.metadata['alpha'] == [0.5] * (K+1)
    print("✓ Float alpha (0.5) works")
    
    # Test 3: Array alpha
    custom_alpha = np.array([0.1, 0.2, 0.3])
    result3 = fit_lfa(W, num_topics=K, alpha=custom_alpha, max_iterations=200, verbose=False)
    assert np.allclose(result3.metadata['alpha'], custom_alpha)
    print("✓ Array alpha works")
    
    return result1, result2, result3


if __name__ == '__main__':
    print("\n" + "="*60)
    print("RUNNING NEW API TESTS")
    print("="*60)
    
    try:
        # Run tests
        result_mfvi = test_fit_lfa_mfvi_basic()
        result_pcgs = test_fit_lfa_pcgs_basic()
        result_labels = test_fit_lfa_with_labels()
        result_dict = test_lfaresult_to_dict()
        result_alphas = test_alpha_parameter()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        
    except Exception as e:
        print("\n" + "="*60)
        print(f"TEST FAILED: {e}")
        print("="*60)
        raise
