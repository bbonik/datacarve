#!/usr/bin/env python3
"""
Example 2: undersampling a randomly generated N-dimensional dataset.

Generates a random dataset where each dimension follows a different random
distribution, then calls the undersampling function to create a balanced
subset across all dimensions. Adjust the number of observations, dimensions,
and the target distribution to experiment.

Author: Vasileios Vonikakis
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from datacarve import undersample_dataset


def generate_random_data(
    total_data: int,
    seed: int,
    redundancy: float = 0.1,
) -> np.ndarray:
    """
    Generate a 1D dataset drawn from a randomly chosen distribution.

    Parameters
    ----------
    total_data : int
        Total number of datapoints (observations) to generate.
    seed : int
        Seed controlling the random generator (both the distribution choice
        and the drawn samples), for reproducibility.
    redundancy : float in [0, 1]
        Fraction of datapoints drawn from a uniform distribution. E.g. with
        redundancy=0.1, 90% of the data follows the random distribution and
        10% is uniform. This guarantees some datapoints cover the full range
        of values, since a random distribution may concentrate its mass in a
        narrow range.

    Returns
    -------
    numpy.ndarray of shape (total_data,)
        Datapoints in [0, 1] drawn from the random distribution.
    """
    rng = np.random.default_rng(seed)

    # split the dataset into distribution-following and uniform parts
    redundancy_data = round(total_data * redundancy)
    distribution_data = total_data - redundancy_data

    # pick one of 7 candidate distributions at random
    distribution_choice = rng.integers(0, 7)

    if distribution_choice == 0:
        data_distr = stats.norm.rvs(
            loc=0, scale=1, size=distribution_data, random_state=rng
        )
    elif distribution_choice == 1:
        data_distr = stats.genpareto.rvs(
            c=-rng.uniform(0, 1), loc=0, scale=1,
            size=distribution_data, random_state=rng,
        )
    elif distribution_choice == 2:
        data_distr = stats.triang.rvs(
            c=rng.uniform(0, 1), loc=0, scale=1,
            size=distribution_data, random_state=rng,
        )
    elif distribution_choice == 3:
        data_distr = stats.anglit.rvs(
            loc=0, scale=1, size=distribution_data, random_state=rng
        )
    elif distribution_choice == 4:
        data_distr = stats.nakagami.rvs(
            nu=rng.uniform(0.1, 5), loc=0, scale=1,
            size=distribution_data, random_state=rng,
        )
    elif distribution_choice == 5:
        data_distr = stats.arcsine.rvs(
            loc=0, scale=1, size=distribution_data, random_state=rng
        )
    else:
        data_distr = stats.argus.rvs(
            chi=rng.uniform(0.1, 5), loc=0, scale=1,
            size=distribution_data, random_state=rng,
        )

    # min-max normalization to [0, 1]
    data_distr = (data_distr - data_distr.min()) / (
        data_distr.max() - data_distr.min()
    )

    # add uniformly distributed datapoints to cover the whole range
    data_redun = stats.uniform.rvs(
        loc=0, scale=1, size=redundancy_data, random_state=rng
    )

    return np.concatenate((data_distr, data_redun))


def main() -> None:
    plt.close("all")

    # generate a random dataset: one random distribution per dimension
    data_observations = 5000  # change accordingly
    data_dimensions = 5  # change accordingly

    data = np.zeros((data_observations, data_dimensions), dtype=float)
    for i in range(data_dimensions):
        data[:, i] = generate_random_data(total_data=data_observations, seed=i)

    # run the undersampling optimization
    indices_to_keep = undersample_dataset(
        data=data,
        data_to_keep=1000,
        target_distribution="uniform",
        bins=10,
        lamda=0.5,
        verbose=True,
        scatterplot_matrix=True,
    )

    data_undersampled = data[indices_to_keep]

    print(f"Original dataset size: {data.shape}")
    print(f"Undersampled dataset size: {data_undersampled.shape}")


if __name__ == "__main__":
    main()
