"""The AAL convergence check - the acceptance test for the simulation engine.

Simulated AAL must agree with the analytic ``sum(rate * mean_loss)``. The tolerance
is the Monte Carlo standard error of the estimator, computed in closed form from the
ELT and the frequency model rather than chosen by hand: a fixed percentage tolerance
would pass a biased engine on a fat-tailed portfolio and fail a correct one on a thin
portfolio, and would not adapt when the example portfolio changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from catagg import aal_standard_error, analytic_aal, simulate, to_ylt
from catagg.metrics import analytic_annual_loss_variance

CASES = [
    pytest.param("poisson", None, id="poisson"),
    pytest.param("negative_binomial", 3.0, id="negative_binomial_k3"),
    pytest.param("negative_binomial", 0.5, id="negative_binomial_k0.5"),
]


def _simulated_aal(
    elt: pd.DataFrame, n_years: int, frequency: str, dispersion: float | None, seed: int
) -> float:
    yelt = simulate(elt, n_years=n_years, frequency=frequency, dispersion=dispersion, seed=seed)
    return float(to_ylt(yelt)["aggregate_loss"].mean())


def test_analytic_aal_is_the_rate_weighted_mean_loss(tiny_elt: pd.DataFrame) -> None:
    expected = float((tiny_elt["rate"] * tiny_elt["mean_loss"]).sum())
    assert analytic_aal(tiny_elt) == pytest.approx(expected)


@pytest.mark.parametrize(("frequency", "dispersion"), CASES)
def test_simulated_aal_converges_to_the_analytic_value(
    example_elt: pd.DataFrame, frequency: str, dispersion: float | None
) -> None:
    n_years = 50_000
    expected = analytic_aal(example_elt)
    standard_error = aal_standard_error(example_elt, n_years, frequency, dispersion)

    # The test must be able to detect a bias worth caring about. If four standard
    # errors is wider than 10% of the AAL, the run is too short to prove anything.
    assert 4.0 * standard_error < 0.10 * expected

    simulated = _simulated_aal(example_elt, n_years, frequency, dispersion, seed=1234)
    assert abs(simulated - expected) < 4.0 * standard_error


@pytest.mark.parametrize(("frequency", "dispersion"), CASES)
def test_aal_is_unbiased_across_independent_replications(
    example_elt: pd.DataFrame, frequency: str, dispersion: float | None
) -> None:
    """One seed can be lucky. The mean z-score over independent runs cannot be."""
    n_years, n_replications = 20_000, 8
    expected = analytic_aal(example_elt)
    standard_error = aal_standard_error(example_elt, n_years, frequency, dispersion)

    z_scores = np.array(
        [
            (_simulated_aal(example_elt, n_years, frequency, dispersion, seed) - expected)
            / standard_error
            for seed in range(n_replications)
        ]
    )
    assert abs(z_scores.mean()) < 3.0 / np.sqrt(n_replications)
    assert np.abs(z_scores).max() < 4.0


@pytest.mark.parametrize(("frequency", "dispersion"), CASES)
def test_annual_loss_variance_matches_the_closed_form(
    example_elt: pd.DataFrame, frequency: str, dispersion: float | None
) -> None:
    """AAL alone cannot detect a frequency bug; the second moment can."""
    n_years = 100_000
    expected = analytic_annual_loss_variance(example_elt, frequency, dispersion)
    yelt = simulate(
        example_elt, n_years=n_years, frequency=frequency, dispersion=dispersion, seed=77
    )
    observed = float(to_ylt(yelt)["aggregate_loss"].var(ddof=1))
    assert observed == pytest.approx(expected, rel=0.10)


def test_clustering_raises_the_annual_loss_variance_but_not_the_aal(
    example_elt: pd.DataFrame,
) -> None:
    """The point of the negative binomial, stated as an assertion."""
    poisson = analytic_annual_loss_variance(example_elt, "poisson")
    clustered = analytic_annual_loss_variance(example_elt, "negative_binomial", 0.5)
    assert clustered > 1.5 * poisson

    n_years = 50_000
    for frequency, dispersion in (("poisson", None), ("negative_binomial", 0.5)):
        simulated = _simulated_aal(example_elt, n_years, frequency, dispersion, seed=5)
        tolerance = 4.0 * aal_standard_error(example_elt, n_years, frequency, dispersion)
        assert abs(simulated - analytic_aal(example_elt)) < tolerance


def test_standard_error_falls_as_the_square_root_of_the_year_count(
    example_elt: pd.DataFrame,
) -> None:
    base = aal_standard_error(example_elt, 10_000)
    assert aal_standard_error(example_elt, 40_000) == pytest.approx(base / 2.0)


def test_aal_standard_error_rejects_a_non_positive_year_count(example_elt: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="n_years"):
        aal_standard_error(example_elt, 0)
