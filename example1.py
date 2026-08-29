#!/usr/bin/env python3
"""
Example 1: undersampling a precomputed 6-dimensional dataset.

Loads a 6-dimensional dataset (11K datapoints) from a .mat file and calls
the undersampling function to create a balanced (uniform) subset of 1000
datapoints across all 6 dimensions. Try other target distributions by
changing the ``target_distribution`` argument to 'gaussian', 'weibull',
'triangular', or a custom array of bin weights.

Author: Vasileios Vonikakis
"""

import matplotlib.pyplot as plt
import scipy.io

from datacarve import undersample_dataset


def main() -> None:
    plt.close("all")

    # load the precomputed 6-dimensional dataset
    data = scipy.io.loadmat("data/DATA_random_6D.mat")["A"]

    indices_to_keep = undersample_dataset(
        data=data,
        data_to_keep=1000,
        target_distribution="uniform",
        bins=10,
        lamda=0.5,
        verbose=True,
        scatterplot_matrix="auto",
    )

    data_undersampled = data[indices_to_keep]

    print(f"Original dataset size: {data.shape}")
    print(f"Undersampled dataset size: {data_undersampled.shape}")


if __name__ == "__main__":
    main()
