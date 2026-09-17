"""Secondary uncertainty: fit exactness, regime assignment, and moment recovery.

The acceptance test for this module is
`test_sampled_moments_recover_the_input_per_event`. Its tolerance is the Monte
Carlo standard error of each estimator, computed from the fit's own analytic
moments rather than chosen by hand, in the same spirit as the AAL convergence
test: a fixed percentage tolerance would be slack for a near-symmetric beta and
impossibly tight for a skewed one.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy import special

from catagg import simulate, swiss_multiperil_elt, validate_elt
from catagg.exceptions import CataggError
from catagg.severity import BetaSeverity, SeverityRegime

# Exposure shared by every constructed row, so that a mean loss reads directly
# as a damage ratio.
EXPOSURE = 1.0e7

# Standard normal deviates allowed before a moment-recovery assertion fails.
# Over 40 independent seeds of the grid below the largest observed deviate was
# under 4, for both the mean and the variance estimator.
Z_TOLERANCE = 5.0

SAMPLES_PER_EVENT = 400_000


def _row(
    event_id: int, mean_ratio: float, std_ratio: float, exposure: float = EXPOSURE
) -> dict[str, Any]:
    """Build one ELT row from a damage ratio and a fraction of the feasible bound.

    ``std_ratio`` is a fraction of the largest standard deviation a distribution
    on ``[0, exposure]`` with this mean can have, so ``1.0`` sits exactly on the
    feasibility bound and ``0.0`` is degenerate.
    """
    mean_loss = mean_ratio * exposure
    largest_std = np.sqrt(mean_loss * (exposure - mean_loss))
    return {
        "event_id": event_id,
        "rate": 0.1,
        "mean_loss": mean_loss,
        "std_loss": std_ratio * largest_std,
        "exposure_value": exposure,
        "peril": "hail",
        "region": "Plateau",
    }


GRID = [(m, f) for m in (0.02, 0.15, 0.5, 0.85) for f in (0.05, 0.5, 0.95)]


@pytest.fixture
def grid_elt() -> pd.DataFrame:
    """Span the interior of the feasible region: four means, three variances each."""
    return pd.DataFrame([_row(i + 1, m, f) for i, (m, f) in enumerate(GRID)])


@pytest.fixture
def regime_elt() -> pd.DataFrame:
    """One row in each regime, in a known order.

    Rows 0-2 are degenerate (zero variance at zero loss, in the interior, and at
    full exposure), rows 3-4 sit on the feasibility bound, row 5 is an ordinary
    interior beta.
    """
    return pd.DataFrame(
        [
            _row(1, 0.0, 0.0),
            _row(2, 0.4, 0.0),
            _row(3, 1.0, 0.0),
            _row(4, 0.25, 1.0),
            _row(5, 0.8, 1.0),
            _row(6, 0.3, 0.6),
        ]
    )


def _central_moment_four(fit: BetaSeverity) -> np.ndarray:
    """Return the fourth central moment of each fitted distribution."""
    first, second, third, fourth = (fit.raw_moment(order) for order in (1, 2, 3, 4))
    return fourth - 4.0 * first * third + 6.0 * first**2 * second - 3.0 * first**4


def _draw_per_event(fit: BetaSeverity, n: int, seed: int) -> np.ndarray:
    """Draw ``n`` losses for every fitted event, shaped ``(n_events, n)``."""
    positions = np.repeat(np.arange(len(fit), dtype=np.int64), n)
    return fit.sample(positions, np.random.default_rng(seed)).reshape(len(fit), n)


# ---------------------------------------------------------------------------
# The fit itself, with no simulation involved
# ---------------------------------------------------------------------------


def test_fit_reproduces_the_elt_moments_exactly(grid_elt: pd.DataFrame) -> None:
    """Method of moments means both moments come back, to machine precision."""
    fit = BetaSeverity.from_elt(grid_elt)
    np.testing.assert_allclose(fit.mean, grid_elt["mean_loss"].to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(fit.std, grid_elt["std_loss"].to_numpy(), rtol=1e-12)


def test_fit_reproduces_the_moments_of_the_example_portfolio() -> None:
    elt = swiss_multiperil_elt(n_events=500, seed=3)
    fit = BetaSeverity.from_elt(elt)
    np.testing.assert_allclose(fit.mean, elt["mean_loss"].to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(fit.std, elt["std_loss"].to_numpy(), rtol=1e-12)


def test_fit_reproduces_the_moments_in_every_regime(regime_elt: pd.DataFrame) -> None:
    """Including the two boundary regimes, where the beta itself does not exist."""
    fit = BetaSeverity.from_elt(regime_elt)
    np.testing.assert_allclose(fit.mean, regime_elt["mean_loss"].to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(fit.std, regime_elt["std_loss"].to_numpy(), rtol=1e-12)


def test_regimes_are_assigned_from_the_variance(regime_elt: pd.DataFrame) -> None:
    fit = BetaSeverity.from_elt(regime_elt)
    expected = [
        SeverityRegime.DEGENERATE,
        SeverityRegime.DEGENERATE,
        SeverityRegime.DEGENERATE,
        SeverityRegime.TWO_POINT,
        SeverityRegime.TWO_POINT,
        SeverityRegime.BETA,
    ]
    assert [SeverityRegime(code) for code in fit.regime] == expected


def test_the_interior_fit_has_no_endpoint_mass(grid_elt: pd.DataFrame) -> None:
    """The beta regime is continuous; its atoms exist only in the limit."""
    fit = BetaSeverity.from_elt(grid_elt)
    assert np.all(fit.regime == SeverityRegime.BETA)
    at_zero, at_exposure = fit.endpoint_masses()
    np.testing.assert_array_equal(at_zero, np.zeros(len(fit)))
    np.testing.assert_array_equal(at_exposure, np.zeros(len(fit)))


def test_endpoint_masses_carry_the_whole_two_point_fit(regime_elt: pd.DataFrame) -> None:
    fit = BetaSeverity.from_elt(regime_elt)
    at_zero, at_exposure = fit.endpoint_masses()
    two_point = fit.regime == SeverityRegime.TWO_POINT

    np.testing.assert_allclose(at_zero[two_point] + at_exposure[two_point], 1.0)
    # Mass at full exposure is the damage ratio, which is what makes the mean come out.
    expected = regime_elt["mean_loss"].to_numpy() / regime_elt["exposure_value"].to_numpy()
    np.testing.assert_allclose(at_exposure[two_point], expected[two_point])


def test_a_degenerate_fit_on_an_endpoint_is_a_unit_mass(regime_elt: pd.DataFrame) -> None:
    """Zero variance at zero loss, or at full exposure, is still an atom."""
    fit = BetaSeverity.from_elt(regime_elt)
    at_zero, at_exposure = fit.endpoint_masses()
    assert at_zero[0] == 1.0
    assert at_zero[1] == 0.0
    assert at_exposure[1] == 0.0
    assert at_exposure[2] == 1.0


@pytest.mark.parametrize("order", [1, 2, 3, 4, 7])
def test_raw_moment_matches_an_independent_beta_implementation(
    grid_elt: pd.DataFrame, order: int
) -> None:
    """Check the recursive product against beta functions, ``B(a + k, b) / B(a, b)``.

    scipy's own ``stats.beta.moment`` is not used as the reference: at order
    seven it loses several percent on the skewed rows of this grid, where the
    log-gamma route and the recursion agree to twelve digits.
    """
    fit = BetaSeverity.from_elt(grid_elt)
    ratio = np.exp(
        special.betaln(fit.alpha + order, fit.beta) - special.betaln(fit.alpha, fit.beta)
    )
    np.testing.assert_allclose(fit.raw_moment(order), ratio * fit.exposure_value**order, rtol=1e-10)


@pytest.mark.parametrize("order", [0, -1, 1.5])
def test_raw_moment_rejects_a_non_positive_integer_order(
    grid_elt: pd.DataFrame, order: object
) -> None:
    with pytest.raises(ValueError, match="order"):
        BetaSeverity.from_elt(grid_elt).raw_moment(order)  # type: ignore[arg-type]


def test_a_variance_inside_the_validator_slack_fits_as_two_point() -> None:
    """`validate_elt` tolerates 1e-9 of relative slack; the fit must not trip on it.

    A row arriving a hair above the feasibility bound has no beta. Sending it to
    the endpoints reproduces its mean exactly and its variance to within that
    same slack, which is the honest answer to an infeasible request.
    """
    row = _row(1, 0.3, 1.0)
    row["std_loss"] *= 1.0 + 4.0e-10
    elt = pd.DataFrame([row])
    validate_elt(elt)

    fit = BetaSeverity.from_elt(elt)
    assert SeverityRegime(fit.regime[0]) is SeverityRegime.TWO_POINT
    np.testing.assert_allclose(fit.mean, row["mean_loss"], rtol=1e-12)
    np.testing.assert_allclose(fit.std, row["std_loss"], rtol=1e-9)


def test_a_vanishing_variance_fits_as_degenerate_rather_than_two_point() -> None:
    """A concentration that overflows is a spike, not a pair of atoms.

    ``std_loss`` here is positive but so small against ``exposure_value`` that
    the scaled variance underflows and the beta concentration becomes infinite.
    Routing that to the endpoints would swap a variance of nearly zero for the
    largest the support allows - the worst possible answer.
    """
    row = _row(1, 0.4, 0.0)
    row["std_loss"] = 1.0e-160 * EXPOSURE
    elt = pd.DataFrame([row])
    validate_elt(elt)

    fit = BetaSeverity.from_elt(elt)
    assert SeverityRegime(fit.regime[0]) is SeverityRegime.DEGENERATE
    np.testing.assert_allclose(fit.mean, row["mean_loss"], rtol=1e-12)

    samples = _draw_per_event(fit, 1_000, seed=0)
    np.testing.assert_array_equal(samples, np.full_like(samples, row["mean_loss"]))


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def test_sampled_moments_recover_the_input_per_event(grid_elt: pd.DataFrame) -> None:
    """The acceptance test for this milestone, per event and per moment.

    The mean and the variance of the draws must each land within
    `Z_TOLERANCE` standard errors of the ELT's own values. The standard errors
    come from the fit's analytic moments: ``std / sqrt(n)`` for the mean, and
    ``sqrt((mu_4 - sigma ** 4) / n)`` for the sample variance.
    """
    fit = BetaSeverity.from_elt(grid_elt)
    target_mean = grid_elt["mean_loss"].to_numpy()
    target_std = grid_elt["std_loss"].to_numpy()

    mean_error = target_std / np.sqrt(SAMPLES_PER_EVENT)
    variance_error = np.sqrt((_central_moment_four(fit) - target_std**4) / SAMPLES_PER_EVENT)

    # The run has to be long enough to prove something. These bounds say the
    # test would catch a mean biased by 2% of a standard deviation, or a
    # variance wrong by 10%.
    assert np.all(Z_TOLERANCE * mean_error < 0.02 * target_std)
    assert np.all(Z_TOLERANCE * variance_error < 0.10 * target_std**2)

    samples = _draw_per_event(fit, SAMPLES_PER_EVENT, seed=20260917)
    np.testing.assert_array_less(
        np.abs(samples.mean(axis=1) - target_mean), Z_TOLERANCE * mean_error
    )
    np.testing.assert_array_less(
        np.abs(samples.var(axis=1, ddof=1) - target_std**2), Z_TOLERANCE * variance_error
    )


def test_samples_never_leave_the_support(regime_elt: pd.DataFrame) -> None:
    fit = BetaSeverity.from_elt(regime_elt)
    samples = _draw_per_event(fit, 50_000, seed=11)
    assert np.all(samples >= 0.0)
    assert np.all(samples <= fit.exposure_value[:, None])


def test_two_point_sampling_uses_only_the_endpoints(regime_elt: pd.DataFrame) -> None:
    """The atoms are real: every draw is exactly zero or exactly the exposure."""
    fit = BetaSeverity.from_elt(regime_elt)
    two_point = np.flatnonzero(fit.regime == SeverityRegime.TWO_POINT)
    n = 200_000
    positions = np.repeat(two_point, n)
    samples = fit.sample(positions, np.random.default_rng(5)).reshape(len(two_point), n)

    exposure = fit.exposure_value[two_point][:, None]
    assert np.all((samples == 0.0) | (samples == exposure))

    _, at_exposure = fit.endpoint_masses()
    observed = (samples == exposure).mean(axis=1)
    expected = at_exposure[two_point]
    standard_error = np.sqrt(expected * (1.0 - expected) / n)
    np.testing.assert_array_less(np.abs(observed - expected), 5.0 * standard_error)


def test_degenerate_sampling_returns_the_mean_with_no_randomness(
    regime_elt: pd.DataFrame,
) -> None:
    """A zero-variance event must not consume the stream or wobble."""
    fit = BetaSeverity.from_elt(regime_elt)
    degenerate = np.flatnonzero(fit.regime == SeverityRegime.DEGENERATE)
    positions = np.repeat(degenerate, 1_000)

    rng = np.random.default_rng(7)
    samples = fit.sample(positions, rng)
    np.testing.assert_array_equal(samples, fit.mean_loss[positions])
    # The generator is untouched, so its next draw is its first.
    assert rng.random() == np.random.default_rng(7).random()


def test_sampling_is_reproducible_and_seed_sensitive(grid_elt: pd.DataFrame) -> None:
    fit = BetaSeverity.from_elt(grid_elt)
    first = _draw_per_event(fit, 2_000, seed=42)
    np.testing.assert_array_equal(first, _draw_per_event(fit, 2_000, seed=42))
    assert not np.array_equal(first, _draw_per_event(fit, 2_000, seed=43))


def test_sample_rejects_positions_outside_the_fit(grid_elt: pd.DataFrame) -> None:
    fit = BetaSeverity.from_elt(grid_elt)
    rng = np.random.default_rng(0)
    for bad in ([len(fit)], [-1]):
        with pytest.raises(CataggError, match="positions"):
            fit.sample(np.array(bad, dtype=np.int64), rng)


def test_sampling_no_occurrences_returns_an_empty_array(grid_elt: pd.DataFrame) -> None:
    fit = BetaSeverity.from_elt(grid_elt)
    samples = fit.sample(np.empty(0, dtype=np.int64), np.random.default_rng(0))
    assert samples.shape == (0,)


def test_an_invalid_elt_is_rejected_by_the_fit() -> None:
    """The fit validates rather than trusting its caller."""
    row = _row(1, 0.3, 1.0)
    row["std_loss"] *= 1.5
    with pytest.raises(Exception, match="std_loss"):
        BetaSeverity.from_elt(pd.DataFrame([row]))


# ---------------------------------------------------------------------------
# Wiring into the engine
# ---------------------------------------------------------------------------


def test_simulate_draws_a_spread_of_losses_for_one_event(grid_elt: pd.DataFrame) -> None:
    """Without secondary uncertainty every occurrence of an event is identical."""
    fixed = simulate(grid_elt, n_years=5_000, seed=1, group_by=None)
    stochastic = simulate(
        grid_elt, n_years=5_000, seed=1, group_by=None, secondary_uncertainty=True
    )
    assert fixed.groupby("event_id")["gross_loss"].nunique().max() == 1
    assert stochastic.groupby("event_id")["gross_loss"].nunique().min() > 1


def test_secondary_uncertainty_leaves_the_frequency_stream_untouched(
    grid_elt: pd.DataFrame,
) -> None:
    """Severity draws from a reserved child seed, so toggling it moves nothing else."""
    columns = ["sim_year", "event_id", "occurrence_index"]
    fixed = simulate(grid_elt, n_years=2_000, seed=8)
    stochastic = simulate(grid_elt, n_years=2_000, seed=8, secondary_uncertainty=True)
    pd.testing.assert_frame_equal(fixed[columns], stochastic[columns])


def test_simulated_losses_stay_within_each_events_exposure() -> None:
    elt = swiss_multiperil_elt(n_events=300, seed=2)
    yelt = simulate(elt, n_years=20_000, seed=4, secondary_uncertainty=True)
    exposure = elt.set_index("event_id")["exposure_value"]
    assert np.all(yelt["gross_loss"].to_numpy() >= 0.0)
    assert np.all(yelt["gross_loss"].to_numpy() <= exposure.loc[yelt["event_id"]].to_numpy())
