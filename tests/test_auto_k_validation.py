"""Validation tests for auto-K selection with known ground truth."""

import numpy as np
from lfa import select_num_topics, simulate_topic_disease_data


def test_auto_k_single_scenario(M, D, true_K, n_replicates=10):
    """Test auto-K selection for a single scenario with known K."""
    print(f"\nTesting: M={M}, D={D}, True K={true_K}")
    print("-" * 60)
    
    results = []
    candidate_topics = [max(2, true_K-1), true_K, true_K+1, min(true_K+2, 10)]
    
    for seed in range(n_replicates):
        # Generate data with known K
        W, beta, theta, z = simulate_topic_disease_data(
            seed=seed, M=M, D=D, K=true_K,
            topic_associated_prob=0.30,
            nontopic_associated_prob=0.01
        )
        
        # Run auto-K selection
        selected_k, cv_results = select_num_topics(
            W, 
            candidate_topics=candidate_topics,
            n_folds=3,  # Faster with 3 folds
            verbose=False
        )
        
        # Store results
        results.append({
            'seed': seed,
            'true_K': true_K,
            'selected_K': selected_k,
            'correct': selected_k == true_K,
            'within_1': abs(selected_k - true_K) <= 1,
            'cv_results': cv_results
        })
        
        print(f"  Replicate {seed+1:2d}: Selected K={selected_k} "
              f"({'✓' if selected_k == true_K else '✗'})")
    
    # Compute statistics
    accuracy = sum(r['correct'] for r in results) / len(results)
    within_1_accuracy = sum(r['within_1'] for r in results) / len(results)
    mean_selected = np.mean([r['selected_K'] for r in results])
    
    print(f"\nResults:")
    print(f"  Exact accuracy: {accuracy*100:.1f}%")
    print(f"  Within ±1: {within_1_accuracy*100:.1f}%")
    print(f"  Mean selected K: {mean_selected:.2f}")
    print(f"  True K: {true_K}")
    
    # Show BIC values from first replicate
    print(f"\nExample BIC values (replicate 0):")
    for cv_res in results[0]['cv_results']:
        marker = " ← SELECTED" if cv_res['K'] == results[0]['selected_K'] else ""
        print(f"  K={cv_res['K']}: BIC={cv_res['mean_bic']:8.2f}, "
              f"Perplexity={cv_res['mean_perplexity']:.4f}{marker}")
    
    return {
        'accuracy': accuracy,
        'within_1_accuracy': within_1_accuracy,
        'mean_selected_K': mean_selected,
        'results': results
    }


def run_comprehensive_validation():
    """Run validation across multiple scenarios."""
    print("=" * 70)
    print("AUTO-K SELECTION VALIDATION")
    print("=" * 70)
    
    scenarios = [
        # (M, D, true_K, description)
        (100, 20, 2, "Easy: K=2, clear separation"),
        (150, 24, 3, "Standard: K=3, moderate size"),
        (120, 18, 4, "Complex: K=4, more topics"),
        (200, 30, 3, "Large: K=3, big dataset"),
    ]
    
    all_results = []
    
    for M, D, true_K, description in scenarios:
        print(f"\n{'='*70}")
        print(f"Scenario: {description}")
        scenario_results = test_auto_k_single_scenario(M, D, true_K, n_replicates=5)
        all_results.append({
            'M': M,
            'D': D,
            'true_K': true_K,
            'description': description,
            **scenario_results
        })
    
    # Overall summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    
    overall_accuracy = np.mean([r['accuracy'] for r in all_results])
    overall_within_1 = np.mean([r['within_1_accuracy'] for r in all_results])
    
    print(f"\nAcross all scenarios:")
    print(f"  Mean exact accuracy: {overall_accuracy*100:.1f}%")
    print(f"  Mean within ±1: {overall_within_1*100:.1f}%")
    
    print(f"\nPer scenario:")
    print(f"{'Scenario':<30} {'Accuracy':>10} {'Within ±1':>12}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['description']:<30} {r['accuracy']*100:>9.1f}% {r['within_1_accuracy']*100:>11.1f}%")
    
    # Assessment
    print("\n" + "=" * 70)
    print("ASSESSMENT")
    print("=" * 70)
    
    if overall_accuracy >= 0.7:
        print("✅ EXCELLENT: Auto-K selection working well (≥70% accuracy)")
        print("   Recommendation: Downgrade warning to informational note")
    elif overall_accuracy >= 0.5:
        print("⚠️  GOOD: Auto-K selection working reasonably (50-70% accuracy)")
        print("   Recommendation: Keep warning, but note it's been validated")
    elif overall_accuracy >= 0.3:
        print("❌ POOR: Auto-K selection weak (30-50% accuracy)")
        print("   Recommendation: Keep strong warning, document limitations")
    else:
        print("❌ BROKEN: Auto-K selection not working (<30% accuracy)")
        print("   Recommendation: Mark as deprecated or remove")
    
    if overall_within_1 >= 0.8:
        print("✅ Most selections within ±1 of true K (good enough for practice)")
    
    return all_results


if __name__ == '__main__':
    results = run_comprehensive_validation()
