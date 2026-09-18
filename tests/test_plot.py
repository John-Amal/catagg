"""The plotting module is thin, so these tests check contracts, not pixels."""

from __future__ import annotations

from collections.abc import Iterator

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from catagg import ep_curve, simulate, swiss_multiperil_elt
from catagg.exceptions import CataggError
from catagg.metrics.ep import EPCurve
from catagg.plot import ep_comparison


@pytest.fixture(scope="module")
def yelt() -> pd.DataFrame:
    elt = swiss_multiperil_elt(n_events=120, seed=0)
    return simulate(elt, n_years=8_000, seed=1, secondary_uncertainty=True)


@pytest.fixture(scope="module")
def aggregate(yelt: pd.DataFrame) -> EPCurve:
    return ep_curve(yelt, kind="AEP")


@pytest.fixture(autouse=True)
def _close_figures() -> Iterator[None]:
    yield
    plt.close("all")


def test_axis_labels_name_the_curve_kind(yelt: pd.DataFrame) -> None:
    for kind in ("OEP", "AEP"):
        ax = ep_comparison({"Gross": ep_curve(yelt, kind=kind)})
        assert kind in ax.get_ylabel()
        assert ax.get_xscale() == "log"
        assert "Return period" in ax.get_xlabel()


def test_the_standard_error_band_is_drawn_by_default(aggregate: EPCurve) -> None:
    """The band is the point of the plot, so its absence should be a choice."""
    with_band = ep_comparison({"Gross": aggregate})
    without = ep_comparison({"Gross": aggregate}, show_standard_error=False)
    assert len(with_band.collections) > len(without.collections)


def test_each_curve_becomes_one_line(aggregate: EPCurve, yelt: pd.DataFrame) -> None:
    ax = ep_comparison({"A": aggregate, "B": ep_curve(yelt, kind="AEP")})
    assert len(ax.lines) == 2
    legend = ax.get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["A", "B"]


def test_a_supplied_axes_is_drawn_on(aggregate: EPCurve) -> None:
    _, ax = plt.subplots()
    assert ep_comparison({"Gross": aggregate}, ax=ax, title="Portfolio") is ax
    assert ax.get_title() == "Portfolio"


def test_mixing_occurrence_and_aggregate_curves_is_refused(yelt: pd.DataFrame) -> None:
    """They do not share a vertical axis, so overlaying them would mislead."""
    with pytest.raises(CataggError, match="same kind"):
        ep_comparison({"O": ep_curve(yelt, "OEP"), "A": ep_curve(yelt, "AEP")})


def test_no_curves_is_refused() -> None:
    with pytest.raises(CataggError, match="at least one"):
        ep_comparison({})


def test_the_default_grid_stops_at_the_shortest_simulation(yelt: pd.DataFrame) -> None:
    """A curve cannot be drawn past the return period its own run supports."""
    short = ep_curve(simulate(swiss_multiperil_elt(n_events=120, seed=0), 2_000, seed=2))
    ax = ep_comparison({"short": short, "long": ep_curve(yelt, kind="OEP")})
    for line in ax.lines:
        assert np.asarray(line.get_xdata()).max() <= short.n_years
