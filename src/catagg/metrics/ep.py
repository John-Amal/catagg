"""Exceedance probability curves, tail metrics, and their Monte Carlo error.

An EP curve is the empirical distribution of a per-year loss, read from the
tail. Two of them matter, and they answer different questions:

- **OEP**, occurrence exceedance probability, is the distribution of the
  *largest single event* in a year. It is what a per-occurrence layer responds
  to.
- **AEP**, aggregate exceedance probability, is the distribution of the *total*
  annual loss. It is what aggregate cover and capital respond to.

Both come from the year loss table, so both are available from one simulation.

Return periods
--------------

A return period ``R`` is the reciprocal of an annual exceedance probability:
``R = 200`` means ``P(loss > x) = 1 / 200``. The loss at that return period is
therefore the ``1 - 1 / R`` quantile of the annual loss.

The quantile uses the Weibull plotting position, ``k / (n + 1)`` for the
``k``-th largest of ``n`` years. That choice is not cosmetic: for a continuous
distribution ``k / (n + 1)`` is unbiased for the exceedance probability of the
``k``-th order statistic, whereas the ``k / n`` convention some vendor tools use
is biased and cannot represent the largest year at all without assigning it a
zero exceedance probability. numpy names the convention explicitly, so
``method="weibull"`` in the code below is the whole of the definition.

Standard error is not optional
------------------------------

Every figure this module returns comes with the Monte Carlo standard error of
the estimate, because a return period read off a simulation is a statistic and
carries sampling error like any other. For a realistic portfolio the RP-200 AEP
estimated from 10,000 simulation years has a standard error of several percent
-- large enough to move a reinsurance decision, and almost never reported.

Two estimators are used, and neither introduces a tuning constant.

**Quantiles.** The empirical CDF at the quantile has standard deviation
``sqrt(p * (1 - p) / n)``. Perturbing the probability level by exactly that
amount and reading the quantile function at both ends gives the sampling spread
of the estimate, mapped through the distribution's own shape::

    se = (Q(u + h) - Q(u - h)) / 2,    u = 1 - p,  h = sqrt(p * (1 - p) / n)

This is the Siddiqui density-quantile estimator with the bandwidth pinned to the
statistically natural scale rather than chosen. It needs no density estimate,
degrades correctly in the far tail where the spacing between order statistics is
wide, and against a 60-run replication study it lands within 15% of the true
sampling standard deviation at 10,000 years and within 2% at 100,000.

**TVaR.** The tail mean over ``k = n * p`` years is a mean, so an obvious
estimator of its error is ``s_tail / sqrt(k)``. That estimator is wrong, and
wrong in the dangerous direction: it treats the exceedance threshold as known
when the threshold is itself estimated from the same sample. Against the same
replication study it understates the true error by between a fifth and a third.
The term ``(1 - p) * (TVaR - VaR) ** 2`` restores what it omits::

    se = sqrt((s_tail**2 + (1 - p) * (tvar - var) ** 2) / k)

which lands within 15% at 10,000 years and within 1% at 100,000.

Both estimators run slightly conservative at short simulations, which is the
right direction for a number whose purpose is to stop someone over-reading a
tail.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from catagg.exceptions import CataggError

__all__ = ["STANDARD_RETURN_PERIODS", "EPCurve", "EPKind", "ReturnPeriods", "ep_curve"]

EPKind = Literal["OEP", "AEP"]

#: A scalar return period, or any sequence of them.
ReturnPeriods: TypeAlias = float | int | Sequence[float] | NDArray[np.float64]

#: The return periods a reinsurance submission is usually quoted at.
STANDARD_RETURN_PERIODS: tuple[int, ...] = (2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000)

_SOURCE_COLUMN: dict[str, str] = {"OEP": "max_occurrence_loss", "AEP": "aggregate_loss"}


def _as_return_periods(return_period: ReturnPeriods) -> NDArray[np.float64]:
    """Coerce a scalar or sequence of return periods to a 1-D float array."""
    values = np.atleast_1d(np.asarray(return_period, dtype=np.float64))
    if values.ndim != 1:
        raise CataggError("return_period must be a scalar or a one-dimensional sequence.")
    return values


@dataclass(frozen=True, eq=False)
class EPCurve:
    """An exceedance probability curve, held as the full simulated distribution.

    Build one with `ep_curve`. The whole per-year loss vector is kept rather
    than a summarised curve, so any return period, tail metric or standard
    error can be read afterwards without re-simulating.

    Attributes
    ----------
    losses
        Per-year loss, sorted ascending, one entry per simulated year including
        years with no events.
    kind
        ``"OEP"`` or ``"AEP"``.
    """

    losses: NDArray[np.float64]
    kind: EPKind

    def __len__(self) -> int:
        """Return the number of simulated years."""
        return int(self.losses.shape[0])

    @property
    def n_years(self) -> int:
        """Number of simulated years behind the curve."""
        return len(self)

    @property
    def mean(self) -> float:
        """Mean per-year loss. For an AEP curve this is the AAL."""
        return float(self.losses.mean())

    @property
    def mean_standard_error(self) -> float:
        """Monte Carlo standard error of `mean`.

        An ordinary standard error of a mean. `catagg.aal_standard_error`
        computes the same quantity in closed form from the ELT, without
        simulating; agreement between the two is a check on both.
        """
        return float(self.losses.std(ddof=1) / np.sqrt(self.n_years))

    def _check_return_periods(self, values: NDArray[np.float64]) -> None:
        """Reject return periods the simulation cannot support.

        Raises
        ------
        CataggError
            If any return period is at or below one, or longer than the
            simulation itself. The upper bound is deliberate: reading a
            10,000-year loss off 10,000 simulated years returns the single
            largest year and dresses one observation up as a return period.
        """
        if np.any(values <= 1.0):
            raise CataggError("return_period must be greater than 1.")
        if np.any(values > self.n_years):
            longest = float(values.max())
            raise CataggError(
                f"return_period {longest:g} exceeds the {self.n_years} simulated years. "
                f"Estimating it needs at least {int(np.ceil(longest))} years; beyond that "
                "the estimate is the largest simulated year, not a return period."
            )

    def _quantile(self, probability: NDArray[np.float64]) -> NDArray[np.float64]:
        """Empirical quantile at the Weibull plotting position."""
        clipped = np.clip(probability, 0.0, 1.0)
        return np.asarray(np.quantile(self.losses, clipped, method="weibull"), dtype=np.float64)

    def at_return_period(
        self, return_period: ReturnPeriods = STANDARD_RETURN_PERIODS
    ) -> pd.DataFrame:
        """Return the loss at one or more return periods, with its standard error.

        Parameters
        ----------
        return_period
            Return period in years, scalar or sequence. Defaults to
            `STANDARD_RETURN_PERIODS`.

        Returns
        -------
        pandas.DataFrame
            Columns ``return_period``, ``exceedance_probability``, ``loss`` and
            ``se``, one row per requested return period.

        Raises
        ------
        CataggError
            If a return period is at or below one, or exceeds the number of
            simulated years.

        Examples
        --------
        >>> import catagg as ca
        >>> elt = ca.swiss_multiperil_elt(n_events=200, seed=0)
        >>> yelt = ca.simulate(elt, n_years=20_000, seed=1)
        >>> curve = ca.ep_curve(yelt, kind="AEP")
        >>> list(curve.at_return_period([100, 200]).columns)
        ['return_period', 'exceedance_probability', 'loss', 'se']
        """
        values = _as_return_periods(return_period)
        self._check_return_periods(values)

        probability = 1.0 / values
        level = 1.0 - probability
        bandwidth = np.sqrt(probability * (1.0 - probability) / self.n_years)
        spread = self._quantile(level + bandwidth) - self._quantile(level - bandwidth)

        return pd.DataFrame(
            {
                "return_period": values,
                "exceedance_probability": probability,
                "loss": self._quantile(level),
                "se": spread / 2.0,
            }
        )

    def tvar(self, return_period: ReturnPeriods = STANDARD_RETURN_PERIODS) -> pd.DataFrame:
        """Return the tail value at risk at one or more return periods.

        TVaR is the mean loss in the years that exceed the return period, so it
        answers what capital regimes actually ask: not "how bad is the 1-in-200
        year" but "how bad is the average year worse than that". It is computed
        as the integrated expected shortfall over the largest ``n / R`` years,
        with a fractional weight on the boundary year so the figure stays
        continuous in the return period.

        Parameters
        ----------
        return_period
            Return period in years, scalar or sequence. Defaults to
            `STANDARD_RETURN_PERIODS`.

        Returns
        -------
        pandas.DataFrame
            Columns ``return_period``, ``exceedance_probability``, ``tvar`` and
            ``se``. The standard error is ``NaN`` where fewer than two
            simulated years exceed the return period.

        Raises
        ------
        CataggError
            If a return period is at or below one, or exceeds the number of
            simulated years.
        """
        values = _as_return_periods(return_period)
        self._check_return_periods(values)

        probability = 1.0 / values
        descending = self.losses[::-1]
        var = self._quantile(1.0 - probability)

        exceeding = probability * self.n_years
        whole = np.floor(exceeding).astype(np.int64)
        cumulative = np.concatenate(([0.0], np.cumsum(descending)))

        # The whole years above the threshold, plus the fraction of the next one
        # that the return period reaches into.
        boundary = descending[np.minimum(whole, self.n_years - 1)]
        total = cumulative[whole] + (exceeding - whole) * boundary
        tail_mean = total / exceeding

        standard_error = np.full(values.shape, np.nan, dtype=np.float64)
        for index, count in enumerate(whole):
            if count < 2:
                continue
            tail = descending[:count]
            tail_variance = float(tail.var(ddof=1))
            threshold_term = (1.0 - probability[index]) * (tail_mean[index] - var[index]) ** 2
            standard_error[index] = np.sqrt((tail_variance + threshold_term) / exceeding[index])

        return pd.DataFrame(
            {
                "return_period": values,
                "exceedance_probability": probability,
                "tvar": tail_mean,
                "se": standard_error,
            }
        )

    def to_frame(self) -> pd.DataFrame:
        """Return the whole curve, one row per simulated year.

        Returns
        -------
        pandas.DataFrame
            Columns ``rank`` (1 for the largest year), ``exceedance_probability``
            at the Weibull plotting position, ``return_period`` and ``loss``,
            ordered from the largest loss down.
        """
        rank = np.arange(1, self.n_years + 1, dtype=np.int64)
        probability = rank / (self.n_years + 1)
        return pd.DataFrame(
            {
                "rank": rank,
                "exceedance_probability": probability,
                "return_period": 1.0 / probability,
                "loss": self.losses[::-1],
            }
        )


def ep_curve(losses: pd.DataFrame, kind: EPKind = "OEP") -> EPCurve:
    """Build an exceedance probability curve from a YELT or a YLT.

    Parameters
    ----------
    losses
        A year event loss table as returned by `catagg.simulate`, or a year
        loss table as returned by `catagg.to_ylt`. A YELT is collapsed first.
    kind
        ``"OEP"`` for the annual maximum occurrence, ``"AEP"`` for the annual
        aggregate.

    Returns
    -------
    EPCurve
        The curve, holding one loss per simulated year.

    Raises
    ------
    CataggError
        If ``kind`` is not ``"OEP"`` or ``"AEP"``, or the frame is neither a
        YELT nor a YLT.

    Examples
    --------
    >>> import catagg as ca
    >>> elt = ca.swiss_multiperil_elt(n_events=200, seed=0)
    >>> yelt = ca.simulate(elt, n_years=10_000, seed=1)
    >>> ca.ep_curve(yelt, kind="AEP").n_years
    10000
    """
    from catagg.simulate.engine import to_ylt

    if kind not in _SOURCE_COLUMN:
        raise CataggError(f"kind must be 'OEP' or 'AEP', got {kind!r}.")

    column = _SOURCE_COLUMN[kind]
    if column in losses.columns:
        ylt = losses
    elif "gross_loss" in losses.columns:
        ylt = to_ylt(losses)
    else:
        raise CataggError(
            "losses must be a YELT (with a gross_loss column) or a YLT (with "
            f"a {column} column); got columns {list(losses.columns)}."
        )

    return EPCurve(losses=np.sort(ylt[column].to_numpy(dtype=np.float64)), kind=kind)
