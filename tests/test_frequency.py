"""Tests for the annual count models.

Tolerances are analytic throughout. A sample mean of ``n`` draws has standard
error ``sqrt(Var / n)``; a sample variance has standard error
``sqrt((mu4 - Var ** 2) / n)`` where ``mu4`` is the fourth central moment. Both
reference values come from the distribution itself, so the tests scale correctly
with the parameters instead of relying on a hand-tuned percentage.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from catagg.frequency import NegativeBinomial, Poisson, resolve_frequency

N_DRAWS = 200_000
Z = 4.0


def _variance_standard_error(fourth_central_moment: float, variance: float, n: int) -> float:
    return float(np.sqrt((fourth_central_moment - variance**2) / n))


@pytest.mark.parametrize("group_rate", [0.3, 5.0, 40.0])
def test_poisson_recovers_mean_and_variance(group_rate: float) -> None:
    rng = np.random.default_rng(11)
    counts = Poisson().sample_annual_counts(group_rate, N_DRAWS, rng)

    mean_se = np.sqrt(group_rate / N_DRAWS)
    assert abs(counts.mean() - group_rate) < Z * mean_se

    fourth = group_rate + 3.0 * group_rate**2
    variance_se = _variance_standard_error(fourth, group_rate, N_DRAWS)
    assert abs(counts.var(ddof=1) - group_rate) < Z * variance_se


@pytest.mark.parametrize(("group_rate", "dispersion"), [(5.0, 2.0), (5.0, 20.0), (0.8, 0.5)])
def test_negative_binomial_recovers_mean_and_variance(group_rate: float, dispersion: float) -> None:
    rng = np.random.default_rng(12)
    model = NegativeBinomial(dispersion=dispersion)
    counts = model.sample_annual_counts(group_rate, N_DRAWS, rng)

    variance = model.annual_count_variance(group_rate)
    assert variance == pytest.approx(group_rate + group_rate**2 / dispersion)

    assert abs(counts.mean() - group_rate) < Z * np.sqrt(variance / N_DRAWS)

    frozen = stats.nbinom(n=dispersion, p=dispersion / (dispersion + group_rate))
    fourth = float((frozen.stats("k") + 3.0) * variance**2)
    assert abs(counts.var(ddof=1) - variance) < Z * _variance_standard_error(
        fourth, variance, N_DRAWS
    )


def test_negative_binomial_is_over_dispersed() -> None:
    """The whole reason the model exists: variance strictly above the mean."""
    model = NegativeBinomial(dispersion=2.0)
    assert model.annual_count_variance(5.0) > 5.0
    assert Poisson().annual_count_variance(5.0) == pytest.approx(5.0)


def test_large_dispersion_converges_to_poisson() -> None:
    group_rate = 5.0
    ratios = [
        NegativeBinomial(dispersion=k).annual_count_variance(group_rate) / group_rate
        for k in (1e2, 1e4, 1e6)
    ]
    assert ratios[0] > ratios[1] > ratios[2] > 1.0
    assert ratios[-1] == pytest.approx(1.0, abs=1e-5)


def test_variance_to_mean_ratio_constructor() -> None:
    group_rate, ratio = 6.0, 2.5
    model = NegativeBinomial.from_variance_to_mean_ratio(ratio, group_rate)
    assert model.annual_count_variance(group_rate) / group_rate == pytest.approx(ratio)


def test_variance_to_mean_ratio_rejects_underdispersion() -> None:
    with pytest.raises(ValueError, match="must exceed 1"):
        NegativeBinomial.from_variance_to_mean_ratio(0.9, 5.0)


@pytest.mark.parametrize("dispersion", [0.0, -1.0, np.nan])
def test_invalid_dispersion_is_rejected(dispersion: float) -> None:
    with pytest.raises(ValueError, match="dispersion"):
        NegativeBinomial(dispersion=dispersion)


def test_sampling_is_reproducible() -> None:
    first = Poisson().sample_annual_counts(3.0, 1_000, np.random.default_rng(5))
    second = Poisson().sample_annual_counts(3.0, 1_000, np.random.default_rng(5))
    np.testing.assert_array_equal(first, second)


def test_counts_are_non_negative_integers() -> None:
    counts = NegativeBinomial(dispersion=1.0).sample_annual_counts(
        2.0, 10_000, np.random.default_rng(1)
    )
    assert counts.dtype == np.dtype("int64")
    assert (counts >= 0).all()


class TestResolveFrequency:
    def test_names(self) -> None:
        assert resolve_frequency("poisson") == Poisson()
        assert resolve_frequency("Negative_Binomial", 3.0) == NegativeBinomial(3.0)
        assert resolve_frequency("nb", 3.0) == NegativeBinomial(3.0)

    def test_instance_passes_through(self) -> None:
        model = NegativeBinomial(4.0)
        assert resolve_frequency(model) is model

    def test_unknown_name(self) -> None:
        with pytest.raises(ValueError, match="Unknown frequency model"):
            resolve_frequency("gamma")

    def test_negative_binomial_requires_dispersion(self) -> None:
        with pytest.raises(ValueError, match="requires dispersion"):
            resolve_frequency("negative_binomial")

    def test_poisson_rejects_dispersion(self) -> None:
        with pytest.raises(ValueError, match="no meaning"):
            resolve_frequency("poisson", 2.0)

    def test_instance_rejects_dispersion(self) -> None:
        with pytest.raises(ValueError, match="ignored"):
            resolve_frequency(Poisson(), 2.0)
