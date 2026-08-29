"""Tests for datacarve.core (undersample_dataset and helpers)."""

import numpy as np
import pytest

from datacarve import undersample_dataset
from datacarve.core import _build_target_pdf, _pairwise_correlation_cost

# common kwargs to keep tests quiet, headless, and fast
QUIET = {"verbose": False, "scatterplot_matrix": False}


@pytest.fixture(scope="module")
def uniform_data():
    """Uniformly distributed data: plenty of redundancy in every bin."""
    rng = np.random.default_rng(0)
    return rng.random((2000, 4))


@pytest.fixture(scope="module")
def skewed_data():
    """Beta-distributed (skewed) data: imbalanced across bins."""
    rng = np.random.default_rng(1)
    return rng.beta(2, 5, size=(2000, 3))


# --------------------------------------------------------------- basic output


class TestBasicOutput:
    def test_mask_shape_and_dtype(self, uniform_data):
        mask = undersample_dataset(uniform_data, data_to_keep=200, **QUIET)
        assert mask.shape == (uniform_data.shape[0],)
        assert mask.dtype == bool

    def test_exact_count_selected(self, uniform_data):
        mask = undersample_dataset(uniform_data, data_to_keep=200, **QUIET)
        assert mask.sum() == 200

    def test_keep_all_datapoints(self, uniform_data):
        small = uniform_data[:50]
        mask = undersample_dataset(small, data_to_keep=50, **QUIET)
        assert mask.sum() == 50
        assert mask.all()

    def test_input_data_not_mutated(self, uniform_data):
        original = uniform_data.copy()
        undersample_dataset(uniform_data, data_to_keep=100, **QUIET)
        np.testing.assert_array_equal(uniform_data, original)


# ---------------------------------------------------- distribution shaping


class TestDistributionShaping:
    def test_uniform_target_balances_bins(self, uniform_data):
        keep, bins = 500, 10
        mask = undersample_dataset(
            uniform_data, data_to_keep=keep, bins=bins,
            target_distribution="uniform", **QUIET,
        )
        subset = uniform_data[mask]
        for dim in range(subset.shape[1]):
            counts, _ = np.histogram(subset[:, dim], bins=bins, range=(0, 1))
            # target is keep/bins per bin; allow small slack
            assert np.all(np.abs(counts - keep / bins) <= 5), (
                f"dimension {dim} not balanced: {counts}"
            )

    def test_gaussian_target_is_center_heavy(self, uniform_data):
        mask = undersample_dataset(
            uniform_data, data_to_keep=400, bins=10,
            target_distribution="gaussian", **QUIET,
        )
        subset = uniform_data[mask]
        counts, _ = np.histogram(subset[:, 0], bins=10, range=(0, 1))
        assert counts[4] + counts[5] > counts[0] + counts[9] + counts[1] + counts[8]

    def test_custom_target_distribution(self):
        # 2D data: with few dimensions, enough datapoints exist whose
        # coordinates land in the target's nonzero bins in *all* dimensions
        # simultaneously, so the target marginals are actually achievable.
        rng = np.random.default_rng(7)
        data = rng.random((2000, 2))
        weights = [0, 0, 1, 2, 4, 4, 2, 1, 0, 0]
        mask = undersample_dataset(
            data, data_to_keep=280, bins=10,
            target_distribution=weights, **QUIET,
        )
        subset = data[mask]
        counts, _ = np.histogram(subset[:, 0], bins=10, range=(0, 1))
        # zero-weight bins should receive (close to) no datapoints
        assert counts[0] + counts[1] + counts[8] + counts[9] <= 4
        # heaviest bins should dominate
        assert counts[4] > counts[2]

    def test_skewed_data_toward_uniform(self, skewed_data):
        """The flagship use case: rebalancing an imbalanced dataset."""
        keep, bins = 300, 10
        mask = undersample_dataset(
            skewed_data, data_to_keep=keep, bins=bins,
            target_distribution="uniform", **QUIET,
        )
        subset = skewed_data[mask]
        # scale exactly as the function does before binning
        lo, hi = skewed_data.min(axis=0), skewed_data.max(axis=0)
        scaled = (subset - lo) / (hi - lo)
        for dim in range(scaled.shape[1]):
            counts, _ = np.histogram(scaled[:, dim], bins=bins, range=(0, 1))
            original, _ = np.histogram(
                (skewed_data[:, dim] - lo[dim]) / (hi[dim] - lo[dim]),
                bins=bins, range=(0, 1),
            )
            # the subset must be flatter than the original distribution
            assert counts.std() < original.std() * (keep / len(skewed_data)) * 2


# ------------------------------------------------------------------- solvers


class TestSolverBackends:
    @pytest.mark.parametrize("solver", ["CBC", "SCIP", "SAT"])
    def test_backend_selects_exact_count(self, skewed_data, solver):
        mask = undersample_dataset(
            skewed_data, data_to_keep=200, solver=solver, **QUIET,
        )
        assert mask.sum() == 200

    def test_unknown_solver_raises(self, uniform_data):
        with pytest.raises(ValueError, match="Unknown solver"):
            undersample_dataset(
                uniform_data, data_to_keep=100, solver="GUROBI", **QUIET,
            )


# ---------------------------------------------------------------- validation


class TestInputValidation:
    def test_rejects_1d_data(self):
        with pytest.raises(ValueError, match="2D"):
            undersample_dataset(np.zeros(100), data_to_keep=10, **QUIET)

    def test_rejects_data_to_keep_too_large(self, uniform_data):
        with pytest.raises(ValueError, match="data_to_keep"):
            undersample_dataset(uniform_data, data_to_keep=10**6, **QUIET)

    def test_rejects_data_to_keep_zero(self, uniform_data):
        with pytest.raises(ValueError, match="data_to_keep"):
            undersample_dataset(uniform_data, data_to_keep=0, **QUIET)

    def test_rejects_bad_bins(self, uniform_data):
        with pytest.raises(ValueError, match="bins"):
            undersample_dataset(uniform_data, data_to_keep=100, bins=1, **QUIET)

    def test_rejects_unknown_distribution_name(self, uniform_data):
        with pytest.raises(ValueError, match="target_distribution"):
            undersample_dataset(
                uniform_data, data_to_keep=100,
                target_distribution="lognormal", **QUIET,
            )


# ------------------------------------------------------------ edge cases


class TestEdgeCases:
    def test_constant_feature_does_not_crash(self):
        rng = np.random.default_rng(2)
        data = rng.random((500, 3))
        data[:, 1] = 0.42  # constant feature
        mask = undersample_dataset(data, data_to_keep=100, **QUIET)
        assert mask.sum() == 100

    def test_no_scaling_mode(self):
        rng = np.random.default_rng(3)
        data = rng.random((500, 3))  # already in [0, 1]
        mask = undersample_dataset(
            data, data_to_keep=100, data_scaling=None, **QUIET,
        )
        assert mask.sum() == 100

    def test_lamda_zero(self, uniform_data):
        mask = undersample_dataset(
            uniform_data, data_to_keep=200, lamda=0.0, **QUIET,
        )
        assert mask.sum() == 200

    def test_non_contiguous_input(self, uniform_data):
        view = uniform_data[::2]  # non-contiguous view
        mask = undersample_dataset(view, data_to_keep=100, **QUIET)
        assert mask.sum() == 100


# ------------------------------------------------------------------ helpers


class TestBuildTargetPdf:
    def test_builtin_names_normalized(self):
        for name in ("uniform", "gaussian", "weibull", "triangular"):
            pdf = _build_target_pdf(name, bins=10)
            assert pdf.shape == (10,)
            assert pdf.sum() == pytest.approx(1.0)
            assert np.all(pdf >= 0)

    def test_uniform_is_flat(self):
        pdf = _build_target_pdf("uniform", bins=8)
        np.testing.assert_allclose(pdf, 1 / 8)

    def test_custom_weights_normalized(self):
        pdf = _build_target_pdf([1, 1, 2], bins=3)
        np.testing.assert_allclose(pdf, [0.25, 0.25, 0.5])

    def test_custom_wrong_length_raises(self):
        with pytest.raises(ValueError, match="shape"):
            _build_target_pdf([1, 2, 3], bins=10)

    def test_custom_negative_weight_raises(self):
        with pytest.raises(ValueError, match=">= 0"):
            _build_target_pdf([1, -1, 1], bins=3)

    def test_custom_zero_mass_raises(self):
        with pytest.raises(ValueError, match="zero total mass"):
            _build_target_pdf([0, 0, 0], bins=3)


class TestPairwiseCorrelationCost:
    def test_matches_naive_loop(self):
        rng = np.random.default_rng(4)
        data = rng.random((50, 5))
        avg = 0.5
        fast = _pairwise_correlation_cost(data, avg)
        d = np.abs(data - avg)
        naive = np.array([
            sum(
                d[k, i] * d[k, j]
                for i in range(5) for j in range(i + 1, 5)
            )
            for k in range(50)
        ])
        np.testing.assert_allclose(fast, naive)

    def test_single_dimension_is_zero(self):
        data = np.random.default_rng(5).random((20, 1))
        np.testing.assert_allclose(
            _pairwise_correlation_cost(data, 0.5), 0.0
        )


# ------------------------------------------------------------------ plotting


class TestPlotting:
    def test_plot_scatter_matrix_smoke(self, uniform_data):
        """Smoke test on a headless backend."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from datacarve import plot_scatter_matrix

        plot_scatter_matrix(uniform_data[:100], title="test")
        plt.close("all")


# ------------------------------------------------- per-dimension targets


class TestPerDimensionTargets:
    def test_different_targets_per_dimension(self):
        rng = np.random.default_rng(10)
        data = rng.random((3000, 2))
        mask = undersample_dataset(
            data, data_to_keep=400, bins=10,
            target_distribution=["uniform", "gaussian"], **QUIET,
        )
        subset = data[mask]
        assert mask.sum() == 400

        # dim 0 should be flat
        counts0, _ = np.histogram(subset[:, 0], bins=10, range=(0, 1))
        assert np.all(np.abs(counts0 - 40) <= 5)

        # dim 1 should be center-heavy
        counts1, _ = np.histogram(subset[:, 1], bins=10, range=(0, 1))
        assert counts1[4] + counts1[5] > counts1[0] + counts1[9] + 20

    def test_mixed_name_and_custom_weights(self):
        rng = np.random.default_rng(11)
        data = rng.random((2000, 2))
        mask = undersample_dataset(
            data, data_to_keep=200, bins=10,
            target_distribution=[
                "uniform",
                [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],  # lower half only
            ],
            **QUIET,
        )
        subset = data[mask]
        counts1, _ = np.histogram(subset[:, 1], bins=10, range=(0, 1))
        # upper-half bins of dim 1 should be (close to) empty
        assert counts1[5:].sum() <= 4

    def test_wrong_number_of_specs_raises(self):
        rng = np.random.default_rng(12)
        data = rng.random((500, 3))
        with pytest.raises(ValueError, match="one spec per"):
            undersample_dataset(
                data, data_to_keep=100,
                target_distribution=["uniform", "gaussian"],  # 2 specs, 3 dims
                **QUIET,
            )

    def test_flat_numeric_list_still_means_single_target(self):
        """Backward compatibility: a flat weight list applies to all dims."""
        rng = np.random.default_rng(13)
        data = rng.random((1000, 3))
        mask = undersample_dataset(
            data, data_to_keep=100, bins=10,
            target_distribution=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            **QUIET,
        )
        assert mask.sum() == 100


# ------------------------------------------------- categorical dimensions


class TestCategoricalDims:
    @pytest.fixture
    def data_with_category(self):
        """Numeric dim + imbalanced 3-value categorical dim (70/20/10%)."""
        rng = np.random.default_rng(20)
        n = 3000
        numeric = rng.random(n)
        category = rng.choice([0.0, 1.0, 2.0], size=n, p=[0.7, 0.2, 0.1])
        return np.column_stack([numeric, category])

    def test_balances_imbalanced_categories(self, data_with_category):
        mask = undersample_dataset(
            data_with_category, data_to_keep=300,
            categorical_dims=[1], **QUIET,
        )
        subset = data_with_category[mask]
        counts = [np.sum(subset[:, 1] == c) for c in (0.0, 1.0, 2.0)]
        # uniform target over 3 categories -> 100 each
        assert np.all(np.abs(np.array(counts) - 100) <= 2), counts

    def test_custom_weights_per_category(self, data_with_category):
        mask = undersample_dataset(
            data_with_category, data_to_keep=300,
            target_distribution=["uniform", [3, 2, 1]],
            categorical_dims=[1], **QUIET,
        )
        subset = data_with_category[mask]
        counts = [np.sum(subset[:, 1] == c) for c in (0.0, 1.0, 2.0)]
        # 3:2:1 target over 300 -> 150/100/50
        assert np.all(np.abs(np.array(counts) - [150, 100, 50]) <= 2), counts

    def test_category_values_need_not_be_contiguous(self):
        rng = np.random.default_rng(21)
        n = 1000
        data = np.column_stack([
            rng.random(n),
            rng.choice([-5.0, 3.5, 100.0], size=n, p=[0.6, 0.3, 0.1]),
        ])
        mask = undersample_dataset(
            data, data_to_keep=150, categorical_dims=[1], **QUIET,
        )
        subset = data[mask]
        counts = [np.sum(subset[:, 1] == c) for c in (-5.0, 3.5, 100.0)]
        assert np.all(np.abs(np.array(counts) - 50) <= 2), counts

    def test_bad_categorical_index_raises(self, data_with_category):
        with pytest.raises(ValueError, match="categorical_dims"):
            undersample_dataset(
                data_with_category, data_to_keep=100,
                categorical_dims=[5], **QUIET,
            )

    def test_categorical_weights_wrong_length_raises(self, data_with_category):
        # dim 1 has 3 categories; 4 weights should fail
        with pytest.raises(ValueError, match="shape"):
            undersample_dataset(
                data_with_category, data_to_keep=100,
                target_distribution=["uniform", [1, 2, 3, 4]],
                categorical_dims=[1], **QUIET,
            )
