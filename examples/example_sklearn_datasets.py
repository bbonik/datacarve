#!/usr/bin/env python3
"""
Explore distributional undersampling on classic scikit-learn datasets.

Loads one of the free sklearn datasets (diabetes, iris, or breast cancer),
plots its scatterplot matrix, and undersamples it to a small balanced subset.

Requires scikit-learn (see requirements.txt).

Author: Vasileios Vonikakis
"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris

from datacarve import plot_scatter_matrix, undersample_dataset


def main() -> None:
    plt.close("all")

    # pick a dataset to explore
    dataset = load_diabetes()
    # dataset = load_iris()
    # dataset = load_breast_cancer()

    data = dataset.data

    plot_scatter_matrix(
        data,
        column_names=list(dataset.feature_names),
        title="Original dataset",
    )

    indices_to_keep = undersample_dataset(
        data=data,
        data_to_keep=20,
        target_distribution="uniform",
        bins=10,
        lamda=0.5,
        verbose=True,
        scatterplot_matrix="auto",
    )

    print(f"Original dataset size: {data.shape}")
    print(f"Undersampled dataset size: {data[indices_to_keep].shape}")


if __name__ == "__main__":
    main()
