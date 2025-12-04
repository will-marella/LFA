"""Latent Factor Allocation (LFA) for Disease Topic Modeling.

A Python package for discovering latent disease topics using Bayesian inference.
Supports both MFVI (fast, approximate) and PCGS (slow, exact) algorithms.

Public API
----------
fit_lfa : function
    Fit an LFA model to binary disease data
    
select_num_topics : function
    Automatically select optimal number of topics via cross-validation
    
LFAResult : class
    Results object containing fitted parameters and analysis methods
    
simulate_topic_disease_data : function
    Generate synthetic disease data for testing

Examples
--------
Basic usage:

>>> from lfa import fit_lfa
>>> import numpy as np
>>> W = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 1]])  # Binary disease matrix
>>> result = fit_lfa(W, num_topics=2, algorithm='mfvi')
>>> print(result.summary())

With simulation:

>>> from lfa import fit_lfa, simulate_topic_disease_data
>>> W, _, _, _ = simulate_topic_disease_data(seed=42, M=100, D=20, K=3)
>>> result = fit_lfa(W, num_topics=3)
"""

from lfa.lfa import fit_lfa, select_num_topics
from lfa._core.results import LFAResult
from lfa.simulation import simulate_topic_disease_data

__all__ = [
    'fit_lfa',
    'select_num_topics',
    'LFAResult',
    'simulate_topic_disease_data',
]

__version__ = '0.2.0'