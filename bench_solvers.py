#!/usr/bin/env python3
"""
Temporary benchmark: compare MILP solver backends on the datacarve problem.

Builds the exact same model (binary selection vars + slack vars,
distribution constraints per dimension/bin) through ortools pywraplp with
different backends, and times them on identical data.
"""

import time

import numpy as np
from ortools.linear_solver import pywraplp

BINS = 10
TIME_LIMIT_SEC = 60


def build_and_solve(backend, data, keep, integer_slacks=False):
    n_obs, n_dim = data.shape
    target_pdf = np.full(BINS, 1.0 / BINS)  # uniform target

    dq = np.digitize(data, bins=np.linspace(0, 1, BINS + 1)) - 1
    dq[dq == BINS] = BINS - 1

    bin_centers = (np.arange(1, BINS + 1) - 0.5) / BINS
    avg = float(np.dot(bin_centers, target_pdf))
    d = np.abs(data - avg)
    v = (d.sum(axis=1) ** 2 - (d**2).sum(axis=1)) / 2.0

    solver = pywraplp.Solver.CreateSolver(backend)
    if solver is None:
        return None
    solver.SetTimeLimit(TIME_LIMIT_SEC * 1000)

    x = [solver.BoolVar(f"x[{i}]") for i in range(n_obs)]
    if integer_slacks:
        s = [solver.IntVar(0, n_obs, f"s[{i}]") for i in range(n_dim * BINS)]
    else:
        s = [solver.NumVar(0, solver.infinity(), f"s[{i}]")
             for i in range(n_dim * BINS)]

    lamda = 0.5
    solver.Minimize(
        solver.Sum(lamda * v[i] * x[i] for i in range(n_obs)) + solver.Sum(s)
    )
    solver.Add(solver.Sum(x) == keep)

    for m in range(n_dim):
        for n in range(BINS):
            b = np.ceil(target_pdf[n] * keep)
            slack = s[m * BINS + n]
            members = np.flatnonzero(dq[:, m] == n)
            count = solver.Sum(x[i] for i in members)
            solver.Add(count - slack <= b)
            solver.Add(-count - slack <= -b)

    t0 = time.perf_counter()
    status = solver.Solve()
    elapsed = time.perf_counter() - t0

    status_name = {
        pywraplp.Solver.OPTIMAL: "OPTIMAL",
        pywraplp.Solver.FEASIBLE: "FEASIBLE",
        pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
        pywraplp.Solver.ABNORMAL: "ABNORMAL",
        pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
    }.get(status, f"code={status}")

    n_selected = sum(
        1 for i in range(n_obs) if x[i].solution_value() > 0.5
    ) if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE) else 0

    obj = solver.Objective().Value() if n_selected else float("nan")
    return status_name, elapsed, obj, n_selected


def main():
    import scipy.io

    # the real bundled dataset: 11K x 6, D5 linearly dependent on D0/D3;
    # this instance previously exhausted CBC's 10s budget
    real = scipy.io.loadmat("data/DATA_random_6D.mat")["A"]
    real = (real - real.min(axis=0)) / (real.max(axis=0) - real.min(axis=0))

    rng = np.random.default_rng(42)
    beta = rng.beta(2, 5, size=(20_000, 6))
    beta = (beta - beta.min(axis=0)) / (beta.max(axis=0) - beta.min(axis=0))

    cases = [
        ("real 6D .mat", real, 1_000),
        ("beta 20Kx6", beta, 2_000),
    ]

    for label, data, keep in cases:
        n_obs, n_dim = data.shape
        print(f"\n=== {label}: N={n_obs}, M={n_dim}, keep={keep} "
              f"(time limit {TIME_LIMIT_SEC}s) ===")
        for backend, int_slacks in [("CBC", False), ("SCIP", False),
                                    ("SAT", True), ("HIGHS", False)]:
            result = build_and_solve(backend, data, keep, int_slacks)
            if result is None:
                print(f"{backend:8s}: backend not available")
                continue
            status, elapsed, obj, n_sel = result
            print(f"{backend:8s}: {status:10s} {elapsed:7.2f}s  "
                  f"objective={obj:12.4f}  selected={n_sel}")


if __name__ == "__main__":
    main()
