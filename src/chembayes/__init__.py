"""
ChemBayes - Bayesian optimization with Gaussian Processes for experimental design.
"""

from .optimizer import create_objective, optimize_experiment

__all__ = ['create_objective', 'optimize_experiment']