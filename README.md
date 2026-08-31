# datacarve

[![CI](https://github.com/bbonik/datacarve/actions/workflows/ci.yml/badge.svg)](https://github.com/bbonik/datacarve/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Carve a balanced subset out of a large dataset — distributional dataset undersampling via MILP optimization.**

```bash
pip install datacarve
```
```python
from datacarve import undersample_dataset

mask = undersample_dataset(data, data_to_keep=1000)  # balanced across ALL dimensions
subset = data[mask]
```

`datacarve` is a Mixed Integer Linear Programming (**MILP**) Python tool for **undersampling a dataset** while enforcing a particular **target distribution** across multiple dimensions. It leverages the (possible) **redundancies** in a large dataset to generate a more **compact** version of it with a specified target distribution across each attribute/dimension, while simultaneously minimizing linear correlations among them. Formerly known as `distributional_dataset_undersampling`.

<img src="https://github.com/bbonik/datacarve/raw/master/assets/example.png" width="900">

## Introduction

Datasets can be highly unbalanced: some values/categories may be over-represented, while others may be under-represented. Such imbalance may have a negative impact on many machine learning techniques: the learning algorithm may be very accurate for the over-represented classes, while exhibiting a very high error for the under-represented ones. **Oversampling** (replicating the under-represented classes) or **undersampling** (reducing the over-represented classes) are two typical approaches to address this problem.

**Balancing a dataset across only 1 attribute is straightforward.** For example, building a face gender classifier (M/F) from an imbalanced dataset should be easy: just undersample the majority class or oversample the minority class.

**Things get really tricky if more than one attribute is involved.** For example, assume that we would like to build the same face gender classifier (M/F), but also achieve a balanced performance across different ages, races and facial expressions. In this case, we have to balance the dataset across 4 attributes (gender, age, race, expressions). Even more, some of these attributes are not categorical, for example age, requiring balancing across different age ranges.

**Undersampling across multiple dimensions is a difficult combinatorial problem.** A datapoint may be majority for attribute A, but minority for attribute B. In the previous example, assume that Male training examples are over-represented, but age ranging from 10-20 years is under-represented. Should you delete a Male datapoint of age 10-20? The answer is not straightforward. It is very difficult to know which datapoints to drop in order to achieve a target distribution *across all attributes*.

## Description

This repository implements an **undersampling MILP-based dataset shaping** technique. The optimization leverages the (possible) *redundancies* in a large dataset to generate a more *compact* version of the original dataset with a specified target distribution across each attribute/dimension, while simultaneously minimizing linear correlations among them.

In summary, given a large dataset and a required target distribution, the MILP optimization creates a compact subset of the original dataset by finding the optimal combination of datapoints that:

1. **Enforces the target distribution across all dimensions.**
2. **Minimizes linear correlations between dimensions.**

As such, this technique can be seen as *complementary to dimensionality reduction*: instead of reducing feature dimensions while maintaining the number of observations, we reduce the number of observations while imposing distributional constraints on the dimensions.

The figure above depicts covariance scatter plots for a 6-dimensional dataset with 11K datapoints. The distribution for each dimension is given by a histogram, while the Pearson correlation rho between dimensions and corresponding p-value (in parentheses) are mentioned for each scatter plot. Dimension 5 (D5) is a linear combination of D0 and D3. Three subsets of 1K datapoints are generated with this data shaping technique, so as to have Uniform, Gaussian and Triangular distributions, while minimizing correlations between different dimensions.

## Applications

Any situation where you need a **subset of fixed size whose attributes follow prescribed distributions, jointly across several attributes**, is a candidate for this technique:

- **Fair & balanced ML evaluation sets.** Build test/benchmark sets that are balanced across multiple sensitive or contextual attributes at once (e.g., gender × age × skin tone × pose for face analysis), so that reported accuracy is not dominated by over-represented groups. The same applies to curating balanced fine-tuning or validation subsets.
- **Dataset debiasing / data-centric AI.** Reshape a skewed training set toward a target distribution instead of collecting new data, leveraging redundancy that is already present in the dataset.
- **Causal inference & epidemiology.** Select a control cohort whose covariate distributions match a treatment group (or any reference population). This generalizes matching approaches such as cardinality matching: the target can be *any* distribution, not just another group's.
- **Survey statistics & market research.** Quota sampling and panel calibration: pick respondents so that the sample matches census demographics across several attributes simultaneously.
- **A/B testing.** Assign experiment groups that are balanced across multiple covariates, rather than relying on randomization alone for small samples.
- **Drug discovery / cheminformatics.** Select compound libraries with desired property distributions (molecular weight, logP, solubility, ...) while minimizing redundancy between correlated properties.
- **Simulation & testing.** Choose a representative, affordable subset of test scenarios (e.g., driving scenarios spanning weather × traffic × speed distributions) when running all of them is too expensive.

## How it compares to other approaches

| Approach | What it does | Limitation this method addresses |
|---|---|---|
| **Random / stratified sampling** | Samples uniformly, or balances strata of *one* attribute. | Cannot jointly balance several attributes; multi-attribute stratification explodes combinatorially and leaves many empty strata. |
| **Class balancing** (e.g., random undersampling, SMOTE in `imbalanced-learn`) | Balances a single categorical label, possibly by synthesizing points. | Single-label only; synthetic points may be unrealistic. This method handles multiple *continuous or categorical* dimensions and only ever selects real datapoints. |
| **Reweighting / calibration** (importance weights, raking) | Keeps all data but assigns weights so that weighted statistics match targets. | The dataset stays large and individual high-weight points dominate; many ML pipelines and human-evaluation settings need an *actual subset*, not weights. |
| **Matching methods** (propensity score, cardinality matching) | Selects a control group whose covariates match a treatment group. | Matches to *another sample's* distribution; here the target is arbitrary (uniform, gaussian, custom), and correlation between attributes is minimized explicitly. |
| **Coreset selection / data pruning** | Selects a subset that preserves model loss or gradient information. | Optimizes for a *model's* training objective, not for interpretable distributional guarantees; typically gives no control over per-attribute histograms. |
| **Greedy / heuristic subset selection** | Iteratively picks points that locally improve balance. | No global guarantee: a point that helps attribute A may hurt attribute B. The MILP reasons about all attributes and all points jointly, and returns a certified optimal (or bounded) solution. |

In short: this method occupies a niche none of the standard tools cover — **exact, jointly multi-attribute, distribution-targeted subset selection of real datapoints**. Its trade-off is scale: one binary decision variable per datapoint means it is practical for datasets up to roughly tens of thousands of observations (increase `max_solver_time_sec` for larger problems).

## Installation

Requires Python 3.10+ (tested with Python 3.12).

```bash
pip install datacarve            # core (solver only)
pip install "datacarve[plot]"    # core + scatterplot matrices
```

Or from source:

```bash
git clone https://github.com/bbonik/datacarve.git
cd datacarve

# create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate

pip install -e ".[plot]"
```

The MILP solver is [Google OR-Tools](https://developers.google.com/optimization) (CBC backend), which is installed automatically — no separate solver installation is needed.

## Quick start

```python
import numpy as np
from datacarve import undersample_dataset

rng = np.random.default_rng(0)
data = rng.random((5000, 4))  # [N observations, M dimensions]

mask = undersample_dataset(
    data=data,
    data_to_keep=500,            # size of the undersampled subset
    target_distribution="uniform",  # 'uniform', 'gaussian', 'weibull', 'triangular'
    bins=10,                     # quantization bins per dimension
    lamda=0.5,                   # weight of the correlation-minimization objective
)

subset = data[mask]  # boolean mask over the original observations
```

You can also pass a **custom target distribution** as an array of bin weights (one weight per bin, automatically normalized):

```python
# triangular-ish custom target over 10 bins
mask = undersample_dataset(
    data=data,
    data_to_keep=500,
    target_distribution=[1, 2, 3, 4, 5, 5, 4, 3, 2, 1],
    bins=10,
)
```

Useful options:

| Parameter | Default | Description |
|---|---|---|
| `data_to_keep` | `1000` | Number of datapoints to keep. |
| `data_scaling` | `'minmax'` | Per-feature scaling to [0, 1]. Use `None` if the data is already scaled. |
| `target_distribution` | `'uniform'` | Built-in name, a custom array of bin weights, or a list with one spec per dimension. See [Per-dimension targets](#per-dimension-targets-and-categorical-attributes). |
| `bins` | `10` | Quantization bins per numeric dimension. Categorical dimensions use one bin per unique value. |
| `categorical_dims` | `None` | Column indices to treat as categorical (one bin per unique value). |
| `lamda` | `0.5` | Balance between distribution matching (`0`) and correlation minimization (`>0`). |
| `solver` | `'CBC'` | MILP solver backend: `'CBC'`, `'SCIP'`, or `'SAT'`. See [Choosing a solver](#choosing-a-solver). |
| `max_solver_time_sec` | `10.0` | Time budget for the MILP solver. Increase for large datasets. |
| `verbose` | `True` | Print progress and solver statistics. |
| `scatterplot_matrix` | `'auto'` | Show scatterplot matrices (auto-disabled for >10 dimensions). |

## Per-dimension targets and categorical attributes

Each dimension can get its **own target distribution** — pass a list with one spec per dimension, mixing built-in names and custom weight arrays:

```python
mask = undersample_dataset(
    data=data,  # shape (N, 3)
    data_to_keep=500,
    target_distribution=["uniform", "gaussian", [1, 2, 3, 4, 5, 5, 4, 3, 2, 1]],
)
```

**Categorical attributes** (e.g. gender, race, class labels) should not be quantized into equal-width bins. Mark them with `categorical_dims` and each unique value becomes its own bin, so `'uniform'` means "equal counts per category":

```python
# column 2 holds a label-encoded category (e.g. 0=A, 1=B, 2=C)
mask = undersample_dataset(
    data=data,
    data_to_keep=300,
    target_distribution=["uniform", "uniform", [3, 2, 1]],  # 3:2:1 over categories
    categorical_dims=[2],
)
```

This is the typical recipe for fairness-style curation: balance the categorical attributes exactly (equal counts per gender/race/label) while shaping the continuous attributes (age, pose, brightness) to a target distribution — all jointly, in one optimization.

## Choosing a solver

All three backends are free, open source, and bundled with OR-Tools — no extra installation needed. They solve the exact same model; they differ in *how* they search, which matters once problems get hard.

| Solver | Best for | Character |
|---|---|---|
| `'CBC'` (default) | Easy to moderate problems | Classic branch-and-bound. Fastest when the problem is not too constrained; if it reports `optimal` within the time budget, stay with it. |
| `'SAT'` (CP-SAT) | Hard instances that hit the time limit | Clause-learning search, multi-core. When the status is `feasible` (time ran out before optimality was proven), it typically finds noticeably *better* subsets than CBC in the same time budget. |
| `'SCIP'` | Medium-hard instances | Modern branch-and-cut. Worth trying when CBC finds a solution quickly but struggles to prove it optimal. |

**Rule of thumb:**

1. Start with the default (`'CBC'`).
2. Check the reported result status (printed when `verbose=True`).
3. If the status is `optimal` — done, no reason to switch.
4. If the status is `feasible` (the time budget ran out), re-run with `solver='SAT'` and/or a larger `max_solver_time_sec`. This is where CP-SAT shines: on a hard 11K-point benchmark instance, all solvers hit a 60s budget, but CP-SAT returned the best subset found.
5. If no solution is found at all, the constraints may be too tight for your data: increase `max_solver_time_sec`, reduce `bins`, or reduce `data_to_keep`.

What makes an instance "hard"? More datapoints, more dimensions, strongly imbalanced data relative to the target (little redundancy to exploit), and correlated dimensions all increase difficulty.

## Examples

Two executed walkthrough notebooks in [`notebooks/`](notebooks/):

- **[Building fair, balanced evaluation sets](notebooks/balanced_evaluation_sets.ipynb)** — carves a 1,000-row eval set from the Adult census data, balanced across sex, race, income and age *simultaneously*, and shows why per-group accuracy numbers become trustworthy.
- **[Survey quota sampling](notebooks/survey_quota_sampling.ipynb)** — selects a quota sample from a skewed respondent panel, hitting census-style age/gender/region targets exactly (fully offline, synthetic data).

Runnable scripts in the [`examples/`](examples/) folder:

- **`example_6d_dataset.py`** — undersamples the bundled 6-dimensional dataset (11K datapoints) down to a uniform 1K subset.
- **`example_random_data.py`** — generates a random N-dimensional dataset (a different random distribution per dimension) and undersamples it.
- **`example_sklearn_datasets.py`** — applies the technique to classic scikit-learn datasets (diabetes, iris, breast cancer). Requires `scikit-learn`.

```bash
python examples/example_6d_dataset.py
```

Solver benchmarks live in [`benchmarks/`](benchmarks/).

## Citations

If you use this code in your research please cite the following papers:

1. [Vonikakis, V., Subramanian, R., Arnfred, J., & Winkler, S. A Probabilistic Approach to People-Centric Photo Selection and Sequencing. IEEE Transactions in Multimedia, 11(19), pp.2609-2624, 2017.](https://www.researchgate.net/publication/316569587_A_Probabilistic_Approach_to_People-Centric_Photo_Selection_and_Sequencing)
2. [V. Vonikakis, R. Subramanian, S. Winkler. Shaping Datasets: Optimal Data Selection for Specific Target Distributions. Proc. ICIP2016, Phoenix, USA, Sept. 25-28, 2016.](http://vintage.winklerbros.net/Publications/icip2016a.pdf)
