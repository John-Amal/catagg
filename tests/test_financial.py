"""Tests for the reinsurance structures and the programme that chains them.

The hand-worked cases use losses in millions so every expected value can be
checked in one's head. The invariant tests run on simulated portfolios.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

import numpy as np
import pandas as pd
import pytest

from catagg import (
    AggregateStopLoss,
    CataggError,
    ExcessOfLoss,
    Programme,
    QuotaShare,
    ep_curve,
    simulate,
    swiss_multiperil_elt,
    to_ylt,
)


def _yelt(years: Sequence[int], losses: Sequence[float], n_years: int) -> pd.DataFrame:
    """Build a YELT whose rows are already in chronological order within each year."""
    frame = pd.DataFrame(
        {
            "sim_year": np.asarray(years, dtype=np.int32),
            "event_id": np.arange(len(losses), dtype=np.int64),
            "gross_loss": np.asarray(losses, dtype=np.float64),
        }
    )
    frame.insert(2, "occurrence_index", frame.groupby("sim_year").cumcount().astype(np.int32))
    frame.attrs["n_years"] = n_years
    return frame


def _ceded(structure: object, losses: Sequence[float]) -> list[float]:
    """Cession of one structure on a single year of losses, in the order given."""
    result = Programme([structure]).apply(_yelt([0] * len(losses), losses, n_years=1))  # type: ignore[list-item]
    return [float(value) for value in result.yelt["ceded_loss"]]


# ---------------------------------------------------------------------------
# Hand-worked cases
# ---------------------------------------------------------------------------


class TestQuotaShare:
    def test_cedes_the_stated_share_of_every_occurrence(self) -> None:
        losses = [10.0, 55.0, 200.0]
        np.testing.assert_allclose(_ceded(QuotaShare(cession=0.2), losses), [2.0, 11.0, 40.0])

    @pytest.mark.parametrize("cession", [-0.1, 1.1, float("nan")])
    def test_cession_outside_the_unit_interval_is_rejected(self, cession: float) -> None:
        with pytest.raises(CataggError, match="cession"):
            QuotaShare(cession=cession)


class TestExcessOfLoss:
    """A 50 xs 50 layer with two reinstatements can pay at most 150 in a year."""

    layer = ExcessOfLoss(attachment=50.0, limit=50.0, reinstatements=2)

    def test_worked_example(self) -> None:
        # Layer losses 30, 50, 50, 40 in order. The first three use 130 of the
        # 150 available, so the fourth recovers only the remaining 20.
        assert _ceded(self.layer, [80.0, 120.0, 200.0, 90.0]) == [30.0, 50.0, 50.0, 20.0]

    def test_order_decides_who_is_cut_off_but_not_the_annual_total(self) -> None:
        # Same four events in reverse. Now the 80 arrives last and is the one cut.
        reversed_order = _ceded(self.layer, [90.0, 200.0, 120.0, 80.0])
        assert reversed_order == [40.0, 50.0, 50.0, 10.0]
        assert sum(reversed_order) == sum(_ceded(self.layer, [80.0, 120.0, 200.0, 90.0])) == 150.0

    def test_no_reinstatements_means_one_limit_per_year(self) -> None:
        single = ExcessOfLoss(attachment=50.0, limit=50.0, reinstatements=0)
        assert _ceded(single, [80.0, 120.0]) == [30.0, 20.0]

    def test_unlimited_reinstatements_pay_every_occurrence_in_full(self) -> None:
        unlimited = ExcessOfLoss(attachment=50.0, limit=50.0, reinstatements=None)
        assert _ceded(unlimited, [80.0, 120.0, 200.0, 90.0, 40.0]) == [30.0, 50.0, 50.0, 40.0, 0.0]

    def test_the_cap_resets_every_year(self) -> None:
        yelt = _yelt([0, 0, 0, 0, 1, 1, 1, 1], [100.0] * 8, n_years=2)
        result = Programme([ExcessOfLoss(attachment=50.0, limit=50.0, reinstatements=1)]).apply(
            yelt
        )
        assert result.yelt["ceded_loss"].tolist() == [50.0, 50.0, 0.0, 0.0] * 2

    @pytest.mark.parametrize(
        ("keyword", "arguments"),
        [
            ("attachment", {"attachment": -1.0, "limit": 50.0, "reinstatements": 0}),
            ("limit", {"attachment": 0.0, "limit": 0.0, "reinstatements": 0}),
            ("reinstatements", {"attachment": 0.0, "limit": 50.0, "reinstatements": -1}),
            ("reinstatements", {"attachment": 0.0, "limit": 50.0, "reinstatements": 1.5}),
        ],
    )
    def test_invalid_terms_are_rejected(self, keyword: str, arguments: dict[str, float]) -> None:
        with pytest.raises(CataggError, match=keyword):
            ExcessOfLoss(**arguments)  # type: ignore[arg-type]


class TestAggregateStopLoss:
    def test_worked_example(self) -> None:
        # Retained losses 60, 70, 90 accumulate to 60, 130, 220. The first 100
        # is retained; the cover pays the next 100, starting part-way through
        # the second occurrence.
        stop_loss = AggregateStopLoss(attachment=100.0, limit=100.0)
        assert _ceded(stop_loss, [60.0, 70.0, 90.0]) == [0.0, 30.0, 70.0]

    def test_an_unlimited_stop_loss_pays_everything_above_the_attachment(self) -> None:
        stop_loss = AggregateStopLoss(attachment=100.0, limit=float("inf"))
        assert _ceded(stop_loss, [60.0, 70.0, 90.0]) == [0.0, 30.0, 90.0]

    def test_an_occurrence_can_straddle_both_ends(self) -> None:
        stop_loss = AggregateStopLoss(attachment=100.0, limit=50.0)
        assert _ceded(stop_loss, [40.0, 200.0, 10.0]) == [0.0, 50.0, 0.0]

    @pytest.mark.parametrize(
        ("keyword", "arguments"),
        [
            ("attachment", {"attachment": -1.0, "limit": 50.0}),
            ("limit", {"attachment": 0.0, "limit": 0.0}),
        ],
    )
    def test_invalid_terms_are_rejected(self, keyword: str, arguments: dict[str, float]) -> None:
        with pytest.raises(CataggError, match=keyword):
            AggregateStopLoss(**arguments)


class TestProgrammeOrder:
    """Each structure protects what the structures before it left retained."""

    losses: ClassVar[list[float]] = [100.0, 150.0]

    def test_quota_share_then_excess_of_loss(self) -> None:
        # The quota share keeps 80 and 120; the 50 xs 50 layer then sees those.
        programme = Programme(
            [QuotaShare(cession=0.25), ExcessOfLoss(attachment=50.0, limit=50.0, reinstatements=2)]
        )
        result = programme.apply(_yelt([0, 0], self.losses, n_years=1))
        np.testing.assert_allclose(result.ceded_by_structure.iloc[:, 0], [25.0, 37.5])
        np.testing.assert_allclose(result.ceded_by_structure.iloc[:, 1], [25.0, 50.0])
        np.testing.assert_allclose(result.yelt["net_loss"], [50.0, 62.5])

    def test_the_reverse_order_gives_a_different_answer(self) -> None:
        # The layer now sees the gross 100 and 150 and pays 50 on each; the
        # quota share takes a quarter of the 50 and 100 left over.
        programme = Programme(
            [ExcessOfLoss(attachment=50.0, limit=50.0, reinstatements=2), QuotaShare(cession=0.25)]
        )
        result = programme.apply(_yelt([0, 0], self.losses, n_years=1))
        np.testing.assert_allclose(result.yelt["net_loss"], [37.5, 75.0])

    def test_structure_columns_are_labelled_in_programme_order(self) -> None:
        programme = Programme(
            [
                QuotaShare(cession=0.2),
                ExcessOfLoss(attachment=50.0, limit=50.0, reinstatements=2),
                AggregateStopLoss(attachment=100.0, limit=100.0),
            ]
        )
        result = programme.apply(_yelt([0], [10.0], n_years=1))
        assert list(result.ceded_by_structure.columns) == [
            "1_quota_share",
            "2_excess_of_loss",
            "3_aggregate_stop_loss",
        ]


# ---------------------------------------------------------------------------
# Invariants on simulated portfolios
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def stressed() -> tuple[pd.DataFrame, Programme]:
    """Build a portfolio and programme in which every limit genuinely binds.

    Rates are scaled up so that years with several large occurrences are common:
    a test that the reinstatement cap is never exceeded proves nothing on a
    portfolio where it is never reached.
    """
    elt = swiss_multiperil_elt(n_events=300, seed=11)
    elt["rate"] *= 20.0
    yelt = simulate(elt, n_years=4_000, secondary_uncertainty=True, seed=12)
    high = float(np.quantile(yelt["gross_loss"], 0.9))
    programme = Programme(
        [
            QuotaShare(cession=0.2),
            ExcessOfLoss(attachment=0.5 * high, limit=0.5 * high, reinstatements=1),
            AggregateStopLoss(attachment=2.0 * high, limit=2.0 * high),
        ]
    )
    return yelt, programme


def test_net_plus_ceded_is_gross_to_the_last_bit(stressed: tuple[pd.DataFrame, Programme]) -> None:
    """The floating-point form of net + ceded = gross.

    ``ceded_loss`` is one subtraction, ``gross - net``, so adding it back to
    ``net`` can differ from ``gross`` by at most one unit in the last place.
    That is the strongest identity any float accounting can offer; "exactly" in
    the bitwise sense fails on a few percent of occurrences for a 20% share.
    """
    yelt, programme = stressed
    result = programme.apply(yelt).yelt
    net = result["net_loss"].to_numpy()
    ceded = result["ceded_loss"].to_numpy()
    gross = result["gross_loss"].to_numpy()
    np.testing.assert_array_max_ulp(net + ceded, gross, maxulp=1)

    annual = {
        column: to_ylt(result, column=column)["aggregate_loss"]
        for column in ("gross_loss", "ceded_loss", "net_loss")
    }
    np.testing.assert_allclose(
        annual["net_loss"] + annual["ceded_loss"], annual["gross_loss"], rtol=1e-12
    )


def test_no_structure_cedes_more_than_it_was_given(
    stressed: tuple[pd.DataFrame, Programme],
) -> None:
    yelt, programme = stressed
    result = programme.apply(yelt)
    by_structure = result.ceded_by_structure.to_numpy()
    retained = result.yelt["gross_loss"].to_numpy()[:, None] - np.cumsum(by_structure, axis=1)
    assert (by_structure >= 0.0).all()
    assert (retained >= -1e-6).all()
    assert (result.yelt["net_loss"] >= 0.0).all()
    np.testing.assert_allclose(
        by_structure.sum(axis=1), result.yelt["ceded_loss"], rtol=1e-12, atol=1e-6
    )


def test_every_limit_binds_and_none_is_exceeded(stressed: tuple[pd.DataFrame, Programme]) -> None:
    yelt, programme = stressed
    result = programme.apply(yelt)
    years = result.yelt["sim_year"].to_numpy()
    xol, stop_loss = programme.structures[1], programme.structures[2]
    assert isinstance(xol, ExcessOfLoss)
    assert isinstance(stop_loss, AggregateStopLoss)

    per_occurrence = result.ceded_by_structure["2_excess_of_loss"].to_numpy()
    assert per_occurrence.max() <= xol.limit
    xol_annual = np.bincount(years, weights=per_occurrence)
    assert xol_annual.max() <= xol.annual_limit * (1.0 + 1e-12)
    assert (xol_annual >= xol.annual_limit * (1.0 - 1e-12)).sum() > 100, "cap never binds"

    stop_annual = np.bincount(years, weights=result.ceded_by_structure["3_aggregate_stop_loss"])
    assert stop_annual.max() <= stop_loss.limit * (1.0 + 1e-12)
    assert (stop_annual >= stop_loss.limit * (1.0 - 1e-12)).sum() > 100, "limit never binds"


def test_annual_totals_do_not_depend_on_occurrence_order(
    stressed: tuple[pd.DataFrame, Programme],
) -> None:
    """Reordering within a year moves cover between occurrences, never in total."""
    yelt, programme = stressed
    rng = np.random.default_rng(0)
    shuffled = yelt.assign(key=rng.random(len(yelt))).sort_values(["sim_year", "key"])
    shuffled["occurrence_index"] = shuffled.groupby("sim_year").cumcount().astype(np.int32)
    shuffled = shuffled.drop(columns="key")
    shuffled.attrs["n_years"] = yelt.attrs["n_years"]

    original = programme.apply(yelt).yelt
    reordered = programme.apply(shuffled).yelt
    for column in ("ceded_loss", "net_loss"):
        np.testing.assert_allclose(
            to_ylt(reordered, column=column)["aggregate_loss"],
            to_ylt(original, column=column)["aggregate_loss"],
            rtol=1e-12,
        )
    # ...while the net largest occurrence, and so the net OEP, does move.
    assert not np.array_equal(
        to_ylt(reordered, column="net_loss")["max_occurrence_loss"],
        to_ylt(original, column="net_loss")["max_occurrence_loss"],
    )


def test_layer_aal_converges_to_its_analytic_value(tiny_elt: pd.DataFrame) -> None:
    """The layer counterpart of the Σ λᵢ μᵢ check.

    Without secondary uncertainty and with unlimited reinstatements, event ``i``
    cedes the fixed amount ``yᵢ = clip(0.8 μᵢ - A, 0, L)`` behind a 20% quota
    share every time it occurs. Annual ceded loss is then compound Poisson with
    mean ``Σ λᵢ yᵢ`` and variance ``Σ λᵢ yᵢ²``, which sets the tolerance.
    """
    attachment, limit, n_years = 2.0e6, 2.0e7, 200_000
    programme = Programme(
        [
            QuotaShare(cession=0.2),
            ExcessOfLoss(attachment=attachment, limit=limit, reinstatements=None),
        ]
    )
    result = programme.apply(simulate(tiny_elt, n_years=n_years, seed=91))
    simulated = np.bincount(
        result.yelt["sim_year"],
        weights=result.ceded_by_structure["2_excess_of_loss"],
        minlength=n_years,
    ).mean()

    rate = tiny_elt["rate"].to_numpy()
    ceded_per_event = np.clip(0.8 * tiny_elt["mean_loss"].to_numpy() - attachment, 0.0, limit)
    expected = float((rate * ceded_per_event).sum())
    standard_error = float(np.sqrt((rate * ceded_per_event**2).sum() / n_years))
    assert (ceded_per_event > 0).sum() >= 2, "the layer must be hit by more than one event"
    assert abs(simulated - expected) < 4.0 * standard_error


# ---------------------------------------------------------------------------
# The result and its inputs
# ---------------------------------------------------------------------------


def test_result_feeds_ep_curve_through_the_column_argument(tiny_elt: pd.DataFrame) -> None:
    yelt = simulate(tiny_elt, n_years=5_000, seed=92)
    result = Programme([QuotaShare(cession=0.3)]).apply(yelt)
    assert result.n_years == 5_000
    assert list(result.yelt.columns) == [
        "sim_year",
        "event_id",
        "occurrence_index",
        "gross_loss",
        "ceded_loss",
        "net_loss",
    ]
    net = ep_curve(result.yelt, kind="AEP", column="net_loss")
    gross = ep_curve(yelt, kind="AEP")
    np.testing.assert_allclose(net.losses, 0.7 * gross.losses, rtol=1e-12)


def test_rows_are_put_in_chronological_order_before_anything_is_ceded() -> None:
    """Row order is not trusted: the programme sorts on (sim_year, occurrence_index)."""
    yelt = _yelt([0, 0, 0, 0], [80.0, 120.0, 200.0, 90.0], n_years=1)
    scrambled = yelt.iloc[[3, 1, 0, 2]]
    scrambled.attrs["n_years"] = 1
    layer = ExcessOfLoss(attachment=50.0, limit=50.0, reinstatements=2)
    result = Programme([layer]).apply(scrambled)
    assert result.yelt["occurrence_index"].tolist() == [0, 1, 2, 3]
    assert result.yelt["ceded_loss"].tolist() == [30.0, 50.0, 50.0, 20.0]


def test_duplicate_occurrence_positions_are_rejected() -> None:
    yelt = _yelt([0, 0], [80.0, 120.0], n_years=1)
    yelt["occurrence_index"] = 0
    with pytest.raises(CataggError, match="occurrence_index"):
        Programme([QuotaShare(cession=0.2)]).apply(yelt)


def test_a_missing_column_is_rejected() -> None:
    yelt = _yelt([0], [80.0], n_years=1).drop(columns="occurrence_index")
    with pytest.raises(CataggError, match="occurrence_index"):
        Programme([QuotaShare(cession=0.2)]).apply(yelt)


def test_an_empty_yelt_gives_an_empty_result() -> None:
    yelt = _yelt([], [], n_years=10)
    result = Programme([QuotaShare(cession=0.2)]).apply(yelt)
    assert result.yelt.empty
    assert result.n_years == 10
    assert (to_ylt(result.yelt, column="net_loss")["aggregate_loss"] == 0.0).all()


def test_a_misbehaving_custom_structure_is_caught() -> None:
    """The programme checks every cession, so a user-written structure cannot break it."""

    class Greedy:
        label = "greedy"

        def cede(self, retained: np.ndarray, years: np.ndarray) -> np.ndarray:
            return 2.0 * retained

    with pytest.raises(CataggError, match="greedy"):
        Programme([Greedy()]).apply(_yelt([0], [80.0], n_years=1))
