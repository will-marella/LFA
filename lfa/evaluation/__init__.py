"""Evaluation utilities for LFA models against ground truth.

This module provides functions for researchers conducting simulation studies
to evaluate parameter recovery against known ground truth.
"""

from lfa.evaluation.evaluate import evaluate_result, compare_algorithms

__all__ = ['evaluate_result', 'compare_algorithms']
