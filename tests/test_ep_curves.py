"""EP curves, tail metrics, and whether the reported standard errors are true.

The central test here is `test_reported_standard_error_matches_replication`. A
standard error is a claim about how much an estimate would move if the
simulation were run again, and the only honest way to check it is to run the
simulation again. Everything else in this file is an anchor: monotonicity,
an analytic Poisson portfolio whose exceedance probabilities are known in
closed form, and agreement with the closed-form AAL error already in the
package.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from catagg import (
    STANDARD_RETURN_PERIODS,
    EPCurve,
    aal_standard_error,
    analytic_aal,
    ep_curve,
    simulate,
    swiss_multiperil_elt,
    to_ylt,
)
from catagg.exceptions import CataggError

REPLICATIONS = 30
REPLICATION_YEARS = 8_000


@pytest.fixture(scope="module")
def example_elt() -> pd.DataFrame:
    return swiss_multiperil_elt(n_events=150, seed=0)


@pytest.fixture(scope="module")
def curve(example_elt: pd.DataFrame) -> EPCurve:
    yelt = simulate(example_elt, n_years=60_000, seed=11, secondary_uncertainty=True)
    return ep_curve(yelt, kind="AEP")


@pytest.fixture(scope="module")
def replications(example_elt: pd.DataFrame) -> dict[str, np.ndarray]:
    """Independent reruns of the same portfolio, for checking the error claims."""
    losses, tails = [], []
    for offset in range(REPLICATIONS):
        yelt = simulate(
            example_elt,
            n_years=REPLICATION_YEARS,
            seed=3_000 + offset,
            secondary_uncertainty=True,
        )
        replicate = ep_curve(yelt, kind="AEP")
        losses.append(replicate.at_return_period(200)["loss"].iloc[0])
        tails.append(replicate.tvar(200)["tvar"].iloc[0])
    return {"loss": np.array(losses), "tvar": np.array(tails)}


# ---------------------------------------------------------------------------
# Do the reported standard errors tell the truth
# ---------------------------------------------------------------------------


def test_reported_standard_error_matches_replication(
    example_elt: pd.DataFrame, replications: dict[str, np.ndarray]
) -> None:
    """A single run's reported error must predict the spread across many runs.

    Both estimators are allowed to run up to 40% conservative and no more than
    15% optimistic. The asymmetry is deliberate: an overstated tail error costs
    a little precision, an understated one is how a return period gets
    over-read.
    """
    yelt = simulate(example_elt, n_years=REPLICATION_YEARS, seed=99, secondary_uncertainty=True)
    single = ep_curve(yelt, kind="AEP")
    reported_loss = single.at_return_period(200)["se"].iloc[0]
    reported_tvar = single.tvar(200)["se"].iloc[0]

    observed_loss = replications["loss"].std(ddof=1)
    observed_tvar = replications["tvar"].std(ddof=1)

    assert 0.85 < reported_loss / observed_loss < 1.40
    assert 0.85 < reported_tvar / observed_tvar < 1.40


def test_the_naive_tail_mean_error_would_understate_tvar(
    example_elt: pd.DataFrame, replications: dict[str, np.ndarray]
) -> None:
    """Guard the correction term against being quietly removed.

    ``s_tail / sqrt(k)`` treats the exceedance threshold as known when it is
    estimated from the same sample. This test pins the size of that mistake, so
    that dropping the ``(1 - p) * (TVaR - VaR) ** 2`` term fails loudly rather
    than shipping a comfortable, wrong number.
    """
    yelt = simulate(example_elt, n_years=REPLICATION_YEARS, seed=99, secondary_uncertainty=True)
    single = ep_curve(yelt, kind="AEP")
    exceedances = int(REPLICATION_YEARS / 200)
    tail = single.losses[::-1][:exceedances]
    naive = tail.std(ddof=1) / np.sqrt(exceedances)

    observed = replications["tvar"].std(ddof=1)
    reported = single.tvar(200)["se"].iloc[0]
    assert naive < 0.85 * observed
    assert naive < reported


def test_standard_error_falls_with_the_square_root_of_simulation_count(
    example_elt: pd.DataFrame,
) -> None:
    """The point of the validation notebook, as an assertion.

    Sixteen times the years should halve the error twice over.
    """
    short = ep_curve(
        simulate(example_elt, n_years=5_000, seed=7, secondary_uncertainty=True), kind="AEP"
    )
    long = ep_curve(
        simulate(example_elt, n_years=80_000, seed=7, secondary_uncertainty=True), kind="AEP"
    )
    ratio = short.at_return_period(200)["se"].iloc[0] / long.at_return_period(200)["se"].iloc[0]
    assert ratio == pytest.approx(4.0, rel=0.35)


def test_curve_mean_agrees_with_the_closed_form_aal(example_elt: pd.DataFrame) -> None:
    """The simulated AEP mean and the analytic AAL are the same quantity."""
    n_years = 40_000
    yelt = simulate(example_elt, n_years=n_years, seed=5, secondary_uncertainty=True)
    built = ep_curve(yelt, kind="AEP")
    expected = analytic_aal(example_elt)
    closed_form = aal_standard_error(example_elt, n_years, secondary_uncertainty=True)

    assert abs(built.mean - expected) < 4.0 * closed_form
    # The curve's own error estimate and the closed form must also agree.
    assert built.mean_standard_error == pytest.approx(closed_form, rel=0.15)


# ---------------------------------------------------------------------------
# An analytic anchor: a one-event portfolio has a Poisson aggregate
# ---------------------------------------------------------------------------


def test_exceedance_probabilities_match_a_poisson_portfolio() -> None:
    """One event, no severity spread: the annual loss is the event count times its loss.

    Then ``P(loss > k * L)`` is exactly the Poisson survival function, with no
    simulation approximation beyond sampling error.
    """
    rate, loss = 2.5, 1.0e6
    elt = pd.DataFrame(
        [
            {
                "event_id": 1,
                "rate": rate,
                "mean_loss": loss,
                "std_loss": 0.0,
                "exposure_value": 1.0e8,
                "peril": "hail",
                "region": "Plateau",
            }
        ]
    )
    n_years = 200_000
    built = ep_curve(simulate(elt, n_years=n_years, seed=3), kind="AEP")

    for count in (0, 1, 2, 4, 6, 9):
        observed = float((built.losses > (count + 0.5) * loss).mean())
        expected = float(stats.poisson.sf(count, rate))
        standard_error = np.sqrt(expected * (1.0 - expected) / n_years)
        assert abs(observed - expected) < 4.0 * standard_error


def test_occurrence_exceedance_is_a_single_loss_for_a_one_event_portfolio() -> None:
    """With one event the annual maximum is either zero or that event's loss."""
    rate, loss = 0.8, 5.0e6
    elt = pd.DataFrame(
        [
            {
                "event_id": 1,
                "rate": rate,
                "mean_loss": loss,
                "std_loss": 0.0,
                "exposure_value": 1.0e8,
                "peril": "hail",
                "region": "Plateau",
            }
        ]
    )
    n_years = 100_000
    built = ep_curve(simulate(elt, n_years=n_years, seed=3), kind="OEP")
    assert set(np.unique(built.losses)) == {0.0, loss}

    expected = 1.0 - float(stats.poisson.pmf(0, rate))
    observed = float((built.losses > 0).mean())
    assert abs(observed - expected) < 4.0 * np.sqrt(expected * (1 - expected) / n_years)


# ---------------------------------------------------------------------------
# Shape and ordering
# ---------------------------------------------------------------------------


def test_occurrence_loss_never_exceeds_aggregate_loss(example_elt: pd.DataFrame) -> None:
    """A single event cannot be worse than the year that contains it."""
    ylt = to_ylt(simulate(example_elt, n_years=20_000, seed=2, secondary_uncertainty=True))
    occurrence = ep_curve(ylt, kind="OEP").at_return_period(STANDARD_RETURN_PERIODS)
    aggregate = ep_curve(ylt, kind="AEP").at_return_period(STANDARD_RETURN_PERIODS)
    assert np.all(occurrence["loss"].to_numpy() <= aggregate["loss"].to_numpy())


def test_loss_increases_with_return_period(curve: EPCurve) -> None:
    table = curve.at_return_period(STANDARD_RETURN_PERIODS)
    assert np.all(np.diff(table["loss"].to_numpy()) > 0)


def test_tvar_is_never_below_the_loss_at_the_same_return_period(curve: EPCurve) -> None:
    """TVaR averages the years beyond the threshold, so it cannot sit below it."""
    periods = [10, 50, 100, 200, 500]
    var = curve.at_return_period(periods)["loss"].to_numpy()
    tail = curve.tvar(periods)["tvar"].to_numpy()
    assert np.all(tail >= var)


def test_tvar_increases_with_return_period(curve: EPCurve) -> None:
    assert np.all(np.diff(curve.tvar(STANDARD_RETURN_PERIODS)["tvar"].to_numpy()) > 0)


def test_a_scalar_return_period_gives_one_row(curve: EPCurve) -> None:
    assert len(curve.at_return_period(200)) == 1
    assert len(curve.tvar(200)) == 1


def test_default_return_periods_are_the_standard_grid(curve: EPCurve) -> None:
    table = curve.at_return_period()
    assert tuple(table["return_period"].astype(int)) == STANDARD_RETURN_PERIODS


def test_to_frame_uses_weibull_plotting_positions(curve: EPCurve) -> None:
    frame = curve.to_frame()
    assert len(frame) == curve.n_years
    # Rank one is the largest year, at exceedance probability 1 / (n + 1).
    assert frame["loss"].iloc[0] == curve.losses.max()
    assert frame["exceedance_probability"].iloc[0] == pytest.approx(1.0 / (curve.n_years + 1))
    assert np.all(np.diff(frame["loss"].to_numpy()) <= 0)
    np.testing.assert_allclose(
        frame["return_period"].to_numpy(), 1.0 / frame["exceedance_probability"].to_numpy()
    )


def test_tvar_standard_error_is_missing_when_the_tail_is_too_short(
    example_elt: pd.DataFrame,
) -> None:
    """Two exceeding years is the minimum a variance can be computed from."""
    built = ep_curve(
        simulate(example_elt, n_years=1_000, seed=4, secondary_uncertainty=True), kind="AEP"
    )
    assert np.isnan(built.tvar(1_000)["se"].iloc[0])
    assert np.isfinite(built.tvar(100)["se"].iloc[0])


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------


def test_a_return_period_longer_than_the_simulation_is_refused(curve: EPCurve) -> None:
    """Reading a 1-in-n loss off n years returns one observation, not a return period."""
    with pytest.raises(CataggError, match="exceeds the"):
        curve.at_return_period(curve.n_years + 1)
    with pytest.raises(CataggError, match="exceeds the"):
        curve.tvar(curve.n_years + 1)


@pytest.mark.parametrize("bad", [1.0, 0.5, 0.0, -10.0])
def test_a_return_period_at_or_below_one_is_refused(curve: EPCurve, bad: float) -> None:
    with pytest.raises(CataggError, match="greater than 1"):
        curve.at_return_period(bad)


def test_an_unknown_curve_kind_is_refused(example_elt: pd.DataFrame) -> None:
    yelt = simulate(example_elt, n_years=100, seed=1)
    with pytest.raises(CataggError, match="OEP"):
        ep_curve(yelt, kind="EP")  # type: ignore[arg-type]


def test_a_frame_that_is_neither_yelt_nor_ylt_is_refused() -> None:
    with pytest.raises(CataggError, match="YELT"):
        ep_curve(pd.DataFrame({"sim_year": [0, 1], "something": [1.0, 2.0]}), kind="AEP")


def test_a_ylt_and_its_yelt_give_the_same_curve(example_elt: pd.DataFrame) -> None:
    yelt = simulate(example_elt, n_years=5_000, seed=6, secondary_uncertainty=True)
    from_yelt = ep_curve(yelt, kind="AEP")
    from_ylt = ep_curve(to_ylt(yelt), kind="AEP")
    np.testing.assert_array_equal(from_yelt.losses, from_ylt.losses)


# ---------------------------------------------------------------------------
# Reading a loss column other than gross_loss
# ---------------------------------------------------------------------------


def _with_net_column(yelt: pd.DataFrame, fraction: float) -> pd.DataFrame:
    frame = yelt.assign(net_loss=fraction * yelt["gross_loss"])
    frame.attrs["n_years"] = yelt.attrs["n_years"]
    return frame


def test_to_ylt_collapses_the_requested_column(tiny_elt: pd.DataFrame) -> None:
    yelt = _with_net_column(simulate(tiny_elt, n_years=5_000, seed=61), 0.25)
    gross = to_ylt(yelt)
    net = to_ylt(yelt, column="net_loss")
    np.testing.assert_allclose(net["aggregate_loss"], 0.25 * gross["aggregate_loss"])
    np.testing.assert_allclose(net["max_occurrence_loss"], 0.25 * gross["max_occurrence_loss"])


def test_ep_curve_reads_the_requested_column(tiny_elt: pd.DataFrame) -> None:
    yelt = _with_net_column(simulate(tiny_elt, n_years=5_000, seed=62), 0.5)
    for kind in ("OEP", "AEP"):
        gross = ep_curve(yelt, kind=kind)
        net = ep_curve(yelt, kind=kind, column="net_loss")
        np.testing.assert_allclose(net.losses, 0.5 * gross.losses)


def test_an_unknown_column_is_named_in_the_error(tiny_elt: pd.DataFrame) -> None:
    yelt = simulate(tiny_elt, n_years=100, seed=63)
    with pytest.raises(CataggError, match="net_loss"):
        ep_curve(yelt, column="net_loss")
    with pytest.raises(CataggError, match="net_loss"):
        to_ylt(yelt, column="net_loss")


def test_a_column_cannot_be_chosen_on_a_ylt(tiny_elt: pd.DataFrame) -> None:
    """A YLT has already collapsed its occurrences, so there is nothing to choose."""
    ylt = to_ylt(simulate(tiny_elt, n_years=100, seed=64))
    with pytest.raises(CataggError, match="YLT"):
        ep_curve(ylt, column="net_loss")
