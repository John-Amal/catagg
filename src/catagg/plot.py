"""Plotting, kept deliberately thin and deliberately optional.

matplotlib is not a runtime dependency of `catagg`. The engine's four
dependencies are numpy, pandas, pyarrow and scipy, and a library that forces a
plotting stack on anyone importing it has made a decision on their behalf.
Install the extra to use this module::

    pip install catagg[plot]

Everything here returns a matplotlib ``Axes``, so the caller keeps control of
the figure, styling and output format.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from catagg.exceptions import CataggError
from catagg.metrics.ep import EPCurve

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.axes import Axes

__all__ = ["ep_comparison"]

_INSTALL_HINT = (
    "Plotting needs matplotlib, which is not a runtime dependency of catagg. "
    "Install it with `pip install catagg[plot]`."
)


def _require_pyplot() -> Any:
    """Import pyplot on demand, with an actionable message when it is absent."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise CataggError(_INSTALL_HINT) from error
    return plt


def _default_return_periods(n_years: int) -> np.ndarray:
    """Log-spaced return periods from 2 years out to the simulation length."""
    return np.unique(np.geomspace(2.0, float(n_years), num=80).round(4))


def ep_comparison(
    curves: Mapping[str, EPCurve],
    *,
    return_periods: np.ndarray | None = None,
    show_standard_error: bool = True,
    ax: Axes | None = None,
    title: str | None = None,
) -> Axes:
    """Plot one or more exceedance probability curves against return period.

    The shaded band is plus or minus one Monte Carlo standard error. It is drawn
    by default because it is the part reviewers have never seen: at the long
    return periods on the right of the plot the band is wide enough to change a
    decision, and a curve drawn as a clean line hides that.

    Parameters
    ----------
    curves
        Mapping of label to `catagg.EPCurve`. All curves must be the same kind,
        since an OEP and an AEP curve do not share a vertical axis.
    return_periods
        Return periods to evaluate. Defaults to a log-spaced grid running from
        two years to the length of the shortest simulation.
    show_standard_error
        Draw the standard error band.
    ax
        Existing axes to draw on. A new figure is created when omitted.
    title
        Axes title.

    Returns
    -------
    matplotlib.axes.Axes
        The axes drawn on.

    Raises
    ------
    CataggError
        If ``curves`` is empty, mixes OEP and AEP, or matplotlib is not
        installed.
    """
    if not curves:
        raise CataggError("ep_comparison needs at least one curve.")

    kinds = {curve.kind for curve in curves.values()}
    if len(kinds) > 1:
        raise CataggError(
            f"All curves must be the same kind; got {sorted(kinds)}. Plot OEP and "
            "AEP on separate axes."
        )
    kind = kinds.pop()

    plt = _require_pyplot()
    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 4.5))

    if return_periods is None:
        return_periods = _default_return_periods(min(c.n_years for c in curves.values()))

    for label, curve in curves.items():
        table = curve.at_return_period(return_periods)
        loss = table["loss"].to_numpy() / 1e6
        (line,) = ax.plot(table["return_period"], loss, label=label, linewidth=1.8)
        if show_standard_error:
            error = table["se"].to_numpy() / 1e6
            ax.fill_between(
                table["return_period"],
                loss - error,
                loss + error,
                color=line.get_color(),
                alpha=0.20,
                linewidth=0,
            )

    ax.set_xscale("log")
    ax.set_xlabel("Return period (years)")
    ax.set_ylabel(f"{kind} loss (millions)")
    ax.grid(visible=True, which="both", alpha=0.25, linewidth=0.6)
    ax.margins(x=0)
    if len(curves) > 1:
        ax.legend(frameon=False)
    if title:
        ax.set_title(title)
    return ax
