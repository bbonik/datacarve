"""
datacarve: distributional dataset undersampling via MILP optimization.

Carve a compact, distribution-shaped subset out of a large dataset.
Given a dataset and a target distribution, datacarve selects the optimal
combination of datapoints that (1) follows the target distribution across
every dimension simultaneously and (2) minimizes linear correlations
between dimensions.

Basic usage:

    >>> from datacarve import undersample_dataset
    >>> mask = undersample_dataset(data, data_to_keep=1000)
    >>> subset = data[mask]
"""

from datacarve.core import plot_scatter_matrix, undersample_dataset

__all__ = ["undersample_dataset", "plot_scatter_matrix"]
__version__ = "0.1.0"
