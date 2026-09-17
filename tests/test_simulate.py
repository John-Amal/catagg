"""Tests for the year loop."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from catagg import CataggError, NegativeBinomial, simulate, to_ylt
from catagg.io.yelt import YELT_COLUMNS


def test_yelt_has_the_declared_schema(tiny_elt: pd.DataFrame) -> None:
    yelt = simulate(tiny_elt, n_years=1_000, seed=0)
    assert list(yelt.columns) == list(YELT_COLUMNS)
    assert yelt["sim_year"].dtype == np.dtype("int32")
    assert yelt["event_id"].dtype == np.dtype("int64")
    assert yelt.attrs["n_years"] == 1_000


def test_sim_years_are_in_range_and_sorted(tiny_elt: pd.DataFrame) -> None:
    yelt = simulate(tiny_elt, n_years=2_000, seed=1)
    years = yelt["sim_year"].to_numpy()
    assert years.min() >= 0
    assert years.max() < 2_000
    assert (np.diff(years) >= 0).all()


def test_occurrence_index_enumerates_each_year_from_zero(tiny_elt: pd.DataFrame) -> None:
    yelt = simulate(tiny_elt, n_years=5_000, seed=2)
    expected = yelt.groupby("sim_year").cumcount().to_numpy()
    np.testing.assert_array_equal(yelt["occurrence_index"].to_numpy(), expected)


def test_every_simulated_event_id_is_in_the_elt(tiny_elt: pd.DataFrame) -> None:
    yelt = simulate(tiny_elt, n_years=5_000, seed=3)
    assert set(yelt["event_id"]).issubset(set(tiny_elt["event_id"]))


def test_gross_loss_matches_the_event_mean_without_secondary_uncertainty(
    tiny_elt: pd.DataFrame,
) -> None:
    yelt = simulate(tiny_elt, n_years=2_000, seed=4)
    lookup = tiny_elt.set_index("event_id")["mean_loss"]
    np.testing.assert_allclose(
        yelt["gross_loss"].to_numpy(), lookup.loc[yelt["event_id"]].to_numpy()
    )


def test_results_do_not_depend_on_chunk_size(example_elt: pd.DataFrame) -> None:
    """Counts are drawn before any uniform, so the chunk boundary must not matter."""
    reference = simulate(example_elt, n_years=3_000, seed=99, chunk_years=100_000)
    for chunk in (1, 97, 1_000):
        pd.testing.assert_frame_equal(
            simulate(example_elt, n_years=3_000, seed=99, chunk_years=chunk), reference
        )


def test_same_seed_reproduces_the_yelt(example_elt: pd.DataFrame) -> None:
    pd.testing.assert_frame_equal(
        simulate(example_elt, n_years=2_000, seed=17),
        simulate(example_elt, n_years=2_000, seed=17),
    )


def test_different_seeds_give_different_yelts(example_elt: pd.DataFrame) -> None:
    first = simulate(example_elt, n_years=2_000, seed=17)
    second = simulate(example_elt, n_years=2_000, seed=18)
    assert len(first) != len(second) or not first.equals(second)


@pytest.mark.parametrize(
    ("frequency", "dispersion"), [("poisson", None), ("negative_binomial", 3.0)]
)
def test_marginal_event_rates_are_recovered(
    tiny_elt: pd.DataFrame, frequency: str, dispersion: float | None
) -> None:
    """Rate-proportional allocation must leave every event's own rate unchanged.

    For event ``i`` in a group with excess-variance coefficient ``c``, the annual
    count has mean ``rate_i`` and variance ``rate_i + c * rate_i ** 2``, so the
    tolerance on the estimated rate is analytic.
    """
    n_years = 200_000
    yelt = simulate(tiny_elt, n_years=n_years, frequency=frequency, dispersion=dispersion, seed=21)
    observed = yelt["event_id"].value_counts().reindex(tiny_elt["event_id"]).fillna(0).to_numpy()

    rate = tiny_elt["rate"].to_numpy()
    coefficient = 0.0 if dispersion is None else 1.0 / dispersion
    standard_error = np.sqrt((rate + coefficient * rate**2) / n_years)
    z_scores = (observed / n_years - rate) / standard_error
    assert np.abs(z_scores).max() < 4.0


def test_negative_binomial_clusters_at_the_group_level(tiny_elt: pd.DataFrame) -> None:
    """Over-dispersion must survive into the annual count, not vanish per event."""
    n_years = 100_000
    hail = tiny_elt[tiny_elt["peril"] == "hail"]
    group_rate = float(hail["rate"].sum())

    poisson = simulate(hail, n_years=n_years, seed=31)
    clustered = simulate(
        hail, n_years=n_years, frequency="negative_binomial", dispersion=0.5, seed=31
    )

    def annual_count_variance(yelt: pd.DataFrame) -> float:
        counts = np.bincount(yelt["sim_year"].to_numpy(), minlength=n_years)
        return float(counts.var(ddof=1))

    assert annual_count_variance(poisson) == pytest.approx(group_rate, rel=0.05)
    expected = NegativeBinomial(0.5).annual_count_variance(group_rate)
    assert annual_count_variance(clustered) == pytest.approx(expected, rel=0.05)
    assert annual_count_variance(clustered) > 2.0 * annual_count_variance(poisson)


def test_grouping_by_none_treats_the_portfolio_as_one_group(tiny_elt: pd.DataFrame) -> None:
    n_years = 100_000
    yelt = simulate(
        tiny_elt,
        n_years=n_years,
        frequency="negative_binomial",
        dispersion=1.0,
        group_by=None,
        seed=41,
    )
    counts = np.bincount(yelt["sim_year"].to_numpy(), minlength=n_years)
    total_rate = float(tiny_elt["rate"].sum())
    expected = NegativeBinomial(1.0).annual_count_variance(total_rate)
    assert counts.var(ddof=1) == pytest.approx(expected, rel=0.05)


def test_unknown_group_column_is_rejected(tiny_elt: pd.DataFrame) -> None:
    with pytest.raises(CataggError, match="group_by"):
        simulate(tiny_elt, n_years=10, group_by="zone")


def test_secondary_uncertainty_is_explicitly_unimplemented(tiny_elt: pd.DataFrame) -> None:
    with pytest.raises(NotImplementedError, match="secondary_uncertainty"):
        simulate(tiny_elt, n_years=10, secondary_uncertainty=True)


@pytest.mark.parametrize("n_years", [0, -1])
def test_non_positive_n_years_is_rejected(tiny_elt: pd.DataFrame, n_years: int) -> None:
    with pytest.raises(ValueError, match="n_years"):
        simulate(tiny_elt, n_years=n_years)


def test_a_portfolio_that_never_fires_produces_an_empty_yelt(tiny_elt: pd.DataFrame) -> None:
    tiny_elt["rate"] = 1e-12
    yelt = simulate(tiny_elt, n_years=100, seed=0)
    assert yelt.empty
    ylt = to_ylt(yelt)
    assert len(ylt) == 100
    assert (ylt["aggregate_loss"] == 0.0).all()


class TestToYlt:
    def test_one_row_per_simulated_year_including_empty_ones(self, tiny_elt: pd.DataFrame) -> None:
        n_years = 20_000
        yelt = simulate(tiny_elt, n_years=n_years, seed=51)
        ylt = to_ylt(yelt)
        assert len(ylt) == n_years
        np.testing.assert_array_equal(ylt["sim_year"].to_numpy(), np.arange(n_years))
        assert (ylt["aggregate_loss"] == 0.0).sum() > 0

    def test_totals_are_preserved(self, tiny_elt: pd.DataFrame) -> None:
        yelt = simulate(tiny_elt, n_years=10_000, seed=52)
        ylt = to_ylt(yelt)
        assert ylt["aggregate_loss"].sum() == pytest.approx(yelt["gross_loss"].sum())

    def test_max_occurrence_never_exceeds_the_aggregate(self, tiny_elt: pd.DataFrame) -> None:
        ylt = to_ylt(simulate(tiny_elt, n_years=10_000, seed=53))
        assert (ylt["max_occurrence_loss"] <= ylt["aggregate_loss"] + 1e-9).all()

    def test_dropping_empty_years_would_inflate_the_aal(self, tiny_elt: pd.DataFrame) -> None:
        """Guards the specific bug to_ylt exists to prevent."""
        n_years = 50_000
        yelt = simulate(tiny_elt, n_years=n_years, seed=54)
        correct = to_ylt(yelt)["aggregate_loss"].mean()
        naive = yelt.groupby("sim_year")["gross_loss"].sum().mean()
        total_rate = float(tiny_elt["rate"].sum())
        assert naive / correct == pytest.approx(1.0 / (1.0 - np.exp(-total_rate)), rel=0.02)

    def test_missing_n_years_is_an_explicit_error(self, tiny_elt: pd.DataFrame) -> None:
        yelt = simulate(tiny_elt, n_years=1_000, seed=55)
        yelt.attrs.clear()
        with pytest.raises(CataggError, match="n_years is unknown"):
            to_ylt(yelt)

    def test_inconsistent_n_years_is_rejected(self, tiny_elt: pd.DataFrame) -> None:
        yelt = simulate(tiny_elt, n_years=1_000, seed=56)
        with pytest.raises(CataggError, match="inconsistent"):
            to_ylt(yelt, n_years=10)


def test_poisson_thinning_is_exact_and_clustering_correlates_events(
    tiny_elt: pd.DataFrame,
) -> None:
    """Pins the allocation design.

    Drawing a group total and allocating occurrences in proportion to rates is
    exactly equivalent to independent per-event Poisson draws, so the sample
    covariance between two events in the same group must be zero under Poisson.
    Under a negative binomial the shared group draw induces
    ``Cov(N_i, N_j) = rate_i * rate_j / k`` - the clustering, made visible.
    """
    n_years = 200_000
    hail = tiny_elt[tiny_elt["peril"] == "hail"]
    first, second = (float(rate) for rate in hail["rate"])

    def per_event_counts(yelt: pd.DataFrame, event_id: int) -> NDArray[np.int64]:
        selected = yelt.loc[yelt["event_id"] == event_id, "sim_year"].to_numpy()
        return np.bincount(selected, minlength=n_years)

    for dispersion in (None, 0.5):
        frequency = "poisson" if dispersion is None else "negative_binomial"
        yelt = simulate(hail, n_years=n_years, frequency=frequency, dispersion=dispersion, seed=71)
        counts = [per_event_counts(yelt, int(event_id)) for event_id in hail["event_id"]]
        covariance = float(np.cov(counts[0], counts[1], ddof=1)[0, 1])

        coefficient = 0.0 if dispersion is None else 1.0 / dispersion
        expected = coefficient * first * second
        variances = [rate + coefficient * rate**2 for rate in (first, second)]
        standard_error = np.sqrt((variances[0] * variances[1] + expected**2) / n_years)
        assert abs(covariance - expected) < 4.0 * standard_error
