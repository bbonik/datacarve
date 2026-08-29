#!/usr/bin/env python3
"""
Distributional dataset undersampling via Mixed Integer Linear Programming.

Given a large dataset, select an optimal subset of datapoints that
(1) follows a target distribution across every dimension simultaneously and
(2) minimizes linear correlations between dimensions.

The optimization is formulated as a MILP and solved with Google OR-Tools
(CBC backend).

If you use this code for research purposes, please cite:

1. Vonikakis, V., Subramanian, R., Arnfred, J., & Winkler, S.
   A Probabilistic Approach to People-Centric Photo Selection and Sequencing.
   IEEE Transactions in Multimedia, 11(19), pp.2609-2624, 2017.
2. Vonikakis, V., Subramanian, R., & Winkler, S.
   Shaping Datasets: Optimal Data Selection for Specific Target Distributions.
   Proc. ICIP2016, Phoenix, USA, Sept. 25-28, 2016.

Author: Vasileios Vonikakis
"""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np
from ortools.linear_solver import pywraplp
from scipy import stats

__all__ = ["undersample_dataset", "plot_scatter_matrix"]


TargetDistribution = Literal["uniform", "gaussian", "weibull", "triangular"]

# Human-readable names for the ortools result status codes.
_RESULT_STATUS = {
    pywraplp.Solver.OPTIMAL: "optimal",
    pywraplp.Solver.FEASIBLE: "feasible",
    pywraplp.Solver.INFEASIBLE: "infeasible",
    pywraplp.Solver.ABNORMAL: "abnormal",
    pywraplp.Solver.NOT_SOLVED: "not solved",
    pywraplp.Solver.UNBOUNDED: "unbounded",
}


def _build_target_pdf(
    target_distribution: TargetDistribution | Sequence[float],
    bins: int,
) -> np.ndarray:
    """
    Build the (normalized) target histogram over the quantization bins.

    Parameters
    ----------
    target_distribution : str or sequence of float
        Either one of the built-in distribution names
        ('uniform', 'gaussian', 'weibull', 'triangular'), or a custom
        sequence of ``bins`` non-negative weights (one per bin).
    bins : int
        Number of quantization bins.

    Returns
    -------
    numpy.ndarray of shape (bins,)
        Target probability mass per bin, normalized to sum to 1.
    """
    x = np.arange(1, bins + 1) - 0.5  # bin centers

    if isinstance(target_distribution, str):
        if target_distribution == "uniform":
            pdf = stats.uniform.pdf(x, loc=0, scale=bins)
        elif target_distribution == "gaussian":
            pdf = stats.norm.pdf(x, loc=bins / 2, scale=1)
        elif target_distribution == "weibull":
            pdf = stats.weibull_min.pdf(x, c=5, loc=2, scale=1)
        elif target_distribution == "triangular":
            pdf = stats.triang.pdf(x, c=0.75, loc=0, scale=bins)
        else:
            raise ValueError(
                f"Unknown target_distribution '{target_distribution}'. "
                "Expected one of: 'uniform', 'gaussian', 'weibull', "
                "'triangular', or a custom array of bin weights."
            )
    else:
        pdf = np.asarray(target_distribution, dtype=float)
        if pdf.shape != (bins,):
            raise ValueError(
                f"Custom target_distribution must have shape ({bins},) to "
                f"match the number of bins, got {pdf.shape}."
            )
        if np.any(pdf < 0):
            raise ValueError("Custom target_distribution weights must be >= 0.")

    total = pdf.sum()
    if total <= 0:
        raise ValueError(
            "Target distribution has zero total mass over the requested bins."
        )
    return pdf / total  # normalize so bin counts sum to ~data_to_keep


def _normalize_target_specs(
    target_distribution: (
        TargetDistribution
        | Sequence[float]
        | Sequence[TargetDistribution | Sequence[float]]
    ),
    n_dimensions: int,
) -> list:
    """
    Expand the ``target_distribution`` argument into one spec per dimension.

    Accepted forms:

    * a single name ('uniform', ...): applied to every dimension;
    * a flat sequence of numbers: a single custom histogram applied to
      every dimension;
    * a sequence of specs (names and/or nested weight sequences), one per
      dimension.

    Returns
    -------
    list of length n_dimensions
        One target spec (str or array-like of weights) per dimension.
    """
    if isinstance(target_distribution, str):
        return [target_distribution] * n_dimensions

    specs = list(target_distribution)

    # a flat sequence of numbers = one custom histogram for all dimensions
    if all(isinstance(el, (int, float, np.integer, np.floating))
           for el in specs):
        return [specs] * n_dimensions

    # otherwise: one spec per dimension
    if len(specs) != n_dimensions:
        raise ValueError(
            f"Per-dimension target_distribution must have one spec per "
            f"dimension ({n_dimensions}), got {len(specs)}."
        )
    return specs


def _pairwise_correlation_cost(
    data: np.ndarray, avg: np.ndarray | float
) -> np.ndarray:
    """
    Per-datapoint cost used to discourage linear correlations (2nd objective).

    For each datapoint k, computes
    ``v[k] = sum over all dimension pairs (i < j) of |x_ki - avg| * |x_kj - avg|``
    which penalizes points that deviate from the expected mean in several
    dimensions at once (such points contribute most to cross-correlation).

    Vectorized using the identity:
    ``sum_{i<j} d_i d_j = ((sum_i d_i)^2 - sum_i d_i^2) / 2``

    Parameters
    ----------
    data : numpy.ndarray of shape (N, M)
        Scaled dataset.
    avg : float or numpy.ndarray of shape (M,)
        Expected mean value of the target distribution, either shared by
        all dimensions or one value per dimension.

    Returns
    -------
    numpy.ndarray of shape (N,)
        Correlation cost per datapoint.
    """
    d = np.abs(data - avg)  # (N, M), avg broadcasts per column
    return (d.sum(axis=1) ** 2 - (d**2).sum(axis=1)) / 2.0


def undersample_dataset(
    data: np.ndarray,
    data_to_keep: int = 1000,
    data_scaling: Literal["minmax"] | None = "minmax",
    target_distribution: (
        TargetDistribution
        | Sequence[float]
        | Sequence[TargetDistribution | Sequence[float]]
    ) = "uniform",
    bins: int = 10,
    categorical_dims: Sequence[int] | None = None,
    lamda: float = 0.5,
    solver: Literal["CBC", "SCIP", "SAT"] = "CBC",
    max_solver_time_sec: float = 10.0,
    verbose: bool = True,
    scatterplot_matrix: bool | Literal["auto"] = "auto",
) -> np.ndarray:
    """
    Undersample a dataset by imposing distributional and correlational
    constraints across its dimensions.

    Runs a mixed integer linear program (MILP) to find the optimal
    combination of ``data_to_keep`` datapoints whose per-dimension histograms
    are as close as possible to the given target distribution, while
    (optionally) minimizing linear correlations between dimensions.

    Parameters
    ----------
    data : numpy.ndarray of shape (N, M)
        Dataset of N observations and M dimensions.
    data_to_keep : int
        Number of datapoints to keep from the original dataset, in [1, N].
    data_scaling : {'minmax', None}
        Scaling applied to each feature before quantization. With None,
        no scaling is applied and the data is expected to lie in [0, 1].
    target_distribution : str, array-like, or sequence of these
        Distribution to enforce on the undersampled dataset. Three forms
        are accepted:

        * a single name ('uniform', 'gaussian', 'weibull', 'triangular'):
          enforced on every dimension. 'uniform' produces a balanced
          dataset;
        * a flat sequence of non-negative weights (one per bin): a single
          custom histogram enforced on every dimension;
        * a sequence with one spec per dimension, mixing names and custom
          weight sequences, e.g. ``['uniform', 'gaussian', [1, 2, 3]]``
          for 3-dimensional data. For categorical dimensions, custom
          weights must have one entry per category.
    bins : int
        Number of bins into which each numeric dimension is quantized for
        the integer program. Categorical dimensions ignore this and use
        one bin per unique value.
    categorical_dims : sequence of int, optional
        Column indices to treat as categorical. Each unique value in such
        a column becomes its own bin, so a 'uniform' target means "equal
        counts per category". Values must be numeric (e.g. label-encoded);
        the encoding order is only used for reporting, not for binning
        width.
    lamda : float
        Balance between the two objectives: distribution matching vs
        correlation minimization. ``lamda=0`` uses only distributional
        constraints; larger values weight correlation minimization more.
    solver : {'CBC', 'SCIP', 'SAT'}
        MILP solver backend (all free and bundled with OR-Tools, no extra
        installation needed). Guidance on choosing:

        * ``'CBC'`` (default) — classic branch-and-bound. Fastest on easy
          and moderately sized problems; a good first choice. If it
          reports 'optimal' within the time budget, there is no reason to
          switch.
        * ``'SAT'`` (CP-SAT) — clause-learning search. On hard instances
          that exhaust the time budget (status 'feasible' instead of
          'optimal'), it typically finds *better* solutions than CBC in
          the same time. Use it when your problem is large or highly
          constrained and solution quality matters more than speed.
          Note: slack variables are modeled as integers for this backend
          (mathematically equivalent for this problem).
        * ``'SCIP'`` — a modern branch-and-cut solver. Worth trying when
          CBC struggles to *prove* optimality on medium-hard instances.

        Rule of thumb: start with 'CBC'; if the result status is
        'feasible' (time limit hit), re-run with 'SAT' and/or a larger
        ``max_solver_time_sec``.
    max_solver_time_sec : float
        Time budget for the MILP solver, in seconds. If the solver proves
        optimality earlier, it returns earlier; otherwise the best
        solution found so far is returned (status 'feasible').
    verbose : bool
        Whether to print progress information.
    scatterplot_matrix : bool or 'auto'
        Whether to display scatterplot matrices of the original and
        undersampled datasets. With 'auto', plots are shown only for
        datasets of 10 or fewer dimensions.

    Returns
    -------
    numpy.ndarray of bool, shape (N,)
        Selection mask over the original observations: True for each
        datapoint kept in the undersampled dataset. If no solution is
        found, a mask of all False is returned.

    Raises
    ------
    ValueError
        If the inputs are inconsistent (wrong shapes, unknown distribution
        name, or infeasible data_to_keep).
    RuntimeError
        If the MILP solver backend cannot be created.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> data = rng.random((5000, 4))
    >>> mask = undersample_dataset(data, data_to_keep=500, verbose=False,
    ...                            scatterplot_matrix=False)
    >>> subset = data[mask]

    Per-dimension targets and a categorical column (index 2):

    >>> mask = undersample_dataset(
    ...     data, data_to_keep=500,
    ...     target_distribution=['uniform', 'gaussian', 'uniform', 'uniform'],
    ...     categorical_dims=[2],
    ...     verbose=False, scatterplot_matrix=False)
    """
    # ------------------------------------------------------ input validation

    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError(f"data must be a 2D array [N, M], got shape {data.shape}")

    n_observations, n_dimensions = data.shape

    if not 1 <= data_to_keep <= n_observations:
        raise ValueError(
            f"data_to_keep must be in [1, {n_observations}], got {data_to_keep}"
        )
    if bins < 2:
        raise ValueError(f"bins must be >= 2, got {bins}")
    if solver not in ("CBC", "SCIP", "SAT"):
        raise ValueError(
            f"Unknown solver '{solver}'. Expected 'CBC', 'SCIP' or 'SAT'."
        )
    solver_backend = solver

    categorical_set = set(int(c) for c in categorical_dims or [])
    if categorical_set and not all(
        0 <= c < n_dimensions for c in categorical_set
    ):
        raise ValueError(
            f"categorical_dims entries must be column indices in "
            f"[0, {n_dimensions - 1}], got {sorted(categorical_set)}."
        )

    if scatterplot_matrix == "auto":
        scatterplot_matrix = n_dimensions <= 10

    # --------------------------------------------------------- data scaling

    if data_scaling == "minmax":
        # min-max normalization per feature -> [0, 1]
        data_min = data.min(axis=0)
        data_max = data.max(axis=0)
        span = data_max - data_min
        span[span == 0] = 1.0  # constant features: avoid division by zero
        data = (data - data_min) / span

    # ----------------------------------------------- quantize data into bins

    if verbose:
        print("\nQuantizing dataset...")

    # numeric dimensions: `bins` equal-width bins over [0, 1];
    # categorical dimensions: one bin per unique value
    data_quantized = np.zeros((n_observations, n_dimensions), dtype=int)
    n_bins_per_dim: list[int] = []
    edges = np.linspace(0, 1, bins + 1)

    for m in range(n_dimensions):
        if m in categorical_set:
            _, codes = np.unique(data[:, m], return_inverse=True)
            data_quantized[:, m] = codes
            n_bins_per_dim.append(int(codes.max()) + 1)
        else:
            # digitize returns indices in [1, bins+1]; shift to [0, bins-1]
            q = np.digitize(data[:, m], bins=edges) - 1
            q[q == bins] = bins - 1  # datapoints exactly at 1.0
            data_quantized[:, m] = q
            n_bins_per_dim.append(bins)

    # ------------------------------------------ target distribution per bin

    specs = _normalize_target_specs(target_distribution, n_dimensions)
    target_pdfs = [
        _build_target_pdf(spec, n_bins_per_dim[m])
        for m, spec in enumerate(specs)
    ]

    str_specs = [s for s in specs if isinstance(s, str)]
    if len(str_specs) == n_dimensions and len(set(str_specs)) == 1:
        target_name = str_specs[0]
    elif isinstance(target_distribution, str):
        target_name = target_distribution
    else:
        target_name = "custom"

    # ---------------------------------------- display original distributions

    if scatterplot_matrix:
        plot_scatter_matrix(
            data,
            title=f"Original dataset ({n_observations} datapoints)",
        )

    # ------------------------------------------------------ build the MILP
    #
    # Variables:
    #   x[0..N-1]                    binary -> 1 if datapoint is selected
    #   s[0..sum(bins_per_dim)-1]    slack per (dimension, bin) cell
    #                                (continuous; integer for the SAT
    #                                backend, which only supports integer
    #                                arithmetic -- equivalent here since
    #                                bin-count deviations are integral)
    #
    # Objective:
    #   minimize  lamda * sum_k v[k] * x[k]  +  sum s
    #   where v[k] is the per-datapoint correlation cost.
    #
    # Constraints:
    #   sum x == data_to_keep
    #   for every (dimension m, bin n) with target count b = pdf_m[n]*keep:
    #       count_selected(m, n) - s[m,n] <= b     (upper slack bound)
    #      -count_selected(m, n) - s[m,n] <= -b    (lower slack bound)
    #   i.e. |count_selected - b| <= s, and s is minimized.

    if verbose:
        print("Filling problem matrices...")

    solver = pywraplp.Solver.CreateSolver(solver_backend)
    if solver is None:
        raise RuntimeError(
            f"Could not create the '{solver_backend}' MILP solver backend."
        )
    solver.SetTimeLimit(int(max_solver_time_sec * 1000))  # milliseconds

    # expected mean of each dimension's target distribution (used by the
    # correlation cost); for categorical dims, categories are mapped to
    # evenly spaced points in [0, 1] for this purpose
    avg = np.empty(n_dimensions)
    data_for_corr = data.copy()
    for m in range(n_dimensions):
        n_bins_m = n_bins_per_dim[m]
        centers = (np.arange(1, n_bins_m + 1) - 0.5) / n_bins_m
        avg[m] = float(np.dot(centers, target_pdfs[m]))
        if m in categorical_set:
            data_for_corr[:, m] = centers[data_quantized[:, m]]

    # 2nd objective: per-datapoint correlation cost (vectorized)
    v = _pairwise_correlation_cost(data_for_corr, avg)

    # slack bookkeeping: dimension m owns slots
    # [slack_offset[m], slack_offset[m+1]) in the slack vector
    slack_offset = np.concatenate([[0], np.cumsum(n_bins_per_dim)])
    n_slacks = int(slack_offset[-1])

    # decision variables (slacks are >= 0 at any optimum, so bound them at 0;
    # the SAT backend requires integer variables, which is equivalent here
    # because bin-count deviations are integral)
    x = [solver.BoolVar(f"x[{i}]") for i in range(n_observations)]
    if solver_backend == "SAT":
        s = [
            solver.IntVar(0, n_observations, f"s[{i}]")
            for i in range(n_slacks)
        ]
    else:
        s = [
            solver.NumVar(0, solver.infinity(), f"s[{i}]")
            for i in range(n_slacks)
        ]

    # objective: correlation cost on selections + sum of slacks
    solver.Minimize(
        solver.Sum(lamda * v[i] * x[i] for i in range(n_observations))
        + solver.Sum(s)
    )

    # equality constraint: exactly data_to_keep datapoints selected
    solver.Add(solver.Sum(x) == data_to_keep)

    # distribution constraints per (dimension, bin)
    total_constraints = n_slacks
    if verbose:
        print(f"Adding constraints [{0:3d}%]", end="")

    k = 0
    for m in range(n_dimensions):
        for n in range(n_bins_per_dim[m]):
            # target count of selected datapoints in this (dimension, bin)
            b = np.ceil(target_pdfs[m][n] * data_to_keep)
            slack = s[slack_offset[m] + n]

            # selected datapoints falling in bin n of dimension m
            members = np.flatnonzero(data_quantized[:, m] == n)
            count = solver.Sum(x[i] for i in members)

            solver.Add(count - slack <= b)  # upper slack bound
            solver.Add(-count - slack <= -b)  # lower slack bound

            k += 1
            if verbose:
                progress = round(k * 100 / total_constraints)
                print(f"\b\b\b\b\b\b[{progress:3d}%]", end="")

    if verbose:
        print(f"\nNumber of variables = {solver.NumVariables()}")
        print(f"Number of constraints = {solver.NumConstraints()}")

    # ------------------------------------------------- solve the MILP

    if verbose:
        print(f"Solving with {solver_backend}...")

    status = solver.Solve()

    if verbose:
        print(f"Result status = {_RESULT_STATUS.get(status, 'unknown')}")
        print(f"Total cost = {solver.Objective().Value()}")
        print(f"Problem solved in {solver.wall_time():f} milliseconds")
        print(f"Problem solved in {solver.iterations()} iterations")
        print(f"Problem solved in {solver.nodes()} branch-and-bound nodes")
        print()

    # ------------------------------------------------- extract the solution

    indx_selected = np.zeros(n_observations, dtype=bool)

    if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        for i in range(n_observations):
            if x[i].solution_value() > 0.5:
                indx_selected[i] = True

    if indx_selected.sum() > 0:
        if scatterplot_matrix:
            plot_scatter_matrix(
                data[indx_selected, :],
                title=(
                    f"Undersampled dataset ({indx_selected.sum()} "
                    f"datapoints) - {target_name}"
                ),
            )
            plot_scatter_matrix(
                data_quantized[indx_selected, :],
                title=(
                    f"Undersampled dataset quantized ({indx_selected.sum()} "
                    f"datapoints) - {target_name}"
                ),
            )
    elif verbose:
        print("No solution was found")

    return indx_selected


def plot_scatter_matrix(
    data: np.ndarray,
    column_names: Sequence[str] | None = None,
    show_correlation: bool = True,
    alpha: float | None = None,
    title: str | None = None,
) -> None:
    """
    Plot a customized scatterplot matrix (based on pandas).

    Parameters
    ----------
    data : numpy.ndarray of shape (N, M)
        Array of datapoints of N observations and M dimensions.
    column_names : sequence of str, optional
        Names of each data dimension. If None, labels D0, D1, ... are
        auto-generated.
    show_correlation : bool
        Whether to annotate the Pearson correlation coefficient for each
        pair of dimensions on the upper triangle of the matrix.
    alpha : float in [0, 1], optional
        Transparency of each datapoint (0 = transparent, 1 = opaque).
        If None, it is adjusted automatically: more transparent for large
        datasets, less transparent for smaller ones.
    title : str, optional
        Title displayed above the scatterplot matrix.
    """
    # imported lazily so the optimizer can run in headless environments
    import matplotlib.pyplot as plt
    import pandas as pd

    plt.style.use("ggplot")

    if column_names is None:
        column_names = [f"D{i}" for i in range(data.shape[1])]

    # auto alpha according to dataset size, clipped to [0.1, 0.7]
    if alpha is None:
        alpha = float(np.clip((5000 - data.shape[0]) / 5000, 0.1, 0.7))

    df = pd.DataFrame(np.asarray(data, dtype=float), columns=list(column_names))
    axes = pd.plotting.scatter_matrix(
        df, alpha=alpha, figsize=(8, 8), diagonal="hist"
    )

    # annotate Pearson correlation coefficients on the upper triangle
    if show_correlation:
        corr = df.corr().to_numpy()
        for i, j in zip(*np.triu_indices_from(axes, k=1)):
            axes[i, j].annotate(
                f"r={corr[i, j]:.3f}",
                (0.7, 0.9),
                xycoords="axes fraction",
                ha="center",
                va="center",
            )

    if title is not None:
        plt.suptitle(title)

    plt.show()
