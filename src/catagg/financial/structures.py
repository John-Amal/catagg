"""Reinsurance structures.

Each structure is told what the cedant still retains on every occurrence, after
the structures ahead of it in the programme, and returns how much of that it
takes. Every cession is non-negative and never exceeds the retained loss it was
offered; `Programme.apply` checks this for every structure, including ones
written outside this module against the `Structure` protocol.

Only loss is modelled. Premium, ceding commission and reinstatement premium are
cash flows beside the loss and do not enter ``net + ceded = gross``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from catagg.exceptions import CataggError
from catagg.financial._accounting import cap_within_years, cumulative_before

__all__ = ["AggregateStopLoss", "ExcessOfLoss", "QuotaShare", "Structure"]


@runtime_checkable
class Structure(Protocol):
    """What `Programme` requires of a reinsurance structure."""

    @property
    def label(self) -> str:
        """Short snake_case name used to label the structure's result column."""
        ...

    def cede(self, retained: NDArray[np.float64], years: NDArray[np.int64]) -> NDArray[np.float64]:
        """Return the amount ceded on each occurrence.

        Parameters
        ----------
        retained
            Loss still retained on each occurrence before this structure.
        years
            Simulated year of each occurrence. Occurrences are sorted by year
            and chronologically within it.

        Returns
        -------
        numpy.ndarray
            Amount ceded, between zero and ``retained``, on each occurrence.
        """
        ...


def _require_finite_non_negative(name: str, value: float) -> None:
    if not (math.isfinite(value) and value >= 0.0):
        raise CataggError(f"{name} must be finite and non-negative, got {value!r}.")


def _require_positive(name: str, value: float) -> None:
    if not value > 0.0:  # also rejects NaN
        raise CataggError(f"{name} must be positive, got {value!r}.")


@dataclass(frozen=True, kw_only=True)
class QuotaShare:
    """Cede a fixed share of every occurrence.

    Parameters
    ----------
    cession
        Share ceded, between 0 and 1.

    Examples
    --------
    >>> import numpy as np
    >>> qs = QuotaShare(cession=0.25)
    >>> qs.cede(np.array([100.0, 40.0]), np.zeros(2, dtype=np.int64)).tolist()
    [25.0, 10.0]
    """

    cession: float
    label: ClassVar[str] = "quota_share"

    def __post_init__(self) -> None:
        if not 0.0 <= self.cession <= 1.0:  # also rejects NaN
            raise CataggError(f"cession must lie in [0, 1], got {self.cession!r}.")

    def cede(self, retained: NDArray[np.float64], years: NDArray[np.int64]) -> NDArray[np.float64]:
        """Cede ``cession`` times each occurrence's retained loss.

        Parameters
        ----------
        retained
            Loss still retained on each occurrence before this structure.
        years
            Simulated year of each occurrence; unused, since a quota share has
            no running total.

        Returns
        -------
        numpy.ndarray
            Amount ceded on each occurrence.
        """
        ceded: NDArray[np.float64] = self.cession * retained
        return ceded


@dataclass(frozen=True, kw_only=True)
class ExcessOfLoss:
    """Per-occurrence excess of loss, with a limited number of reinstatements.

    On each occurrence the layer pays the retained loss above ``attachment``,
    up to ``limit``. Every payment uses up cover, which is reinstated up to
    ``reinstatements`` times, so the most the layer pays in a year is
    ``limit * (1 + reinstatements)``. Cover is consumed in chronological order:
    once the year's cover is gone, later occurrences recover nothing.

    Parameters
    ----------
    attachment
        Retention per occurrence.
    limit
        Most the layer pays on one occurrence.
    reinstatements
        Number of times the limit is reinstated in a year, or ``None`` for
        unlimited reinstatement. Required: market terms vary too widely for any
        default to be safe.

    Notes
    -----
    The order of occurrences changes which one is cut off when the annual cap
    binds, and so the net loss per occurrence and the net OEP. It never changes
    the layer's annual total, which is
    ``min(sum of per-occurrence layer losses, limit * (1 + reinstatements))``.

    Reinstatement premium is not modelled in this version.

    Examples
    --------
    A 50 xs 50 layer with two reinstatements pays at most 150 a year. Of four
    layer losses of 30, 50, 50 and 40, the last recovers only the 20 left:

    >>> import numpy as np
    >>> layer = ExcessOfLoss(attachment=50.0, limit=50.0, reinstatements=2)
    >>> layer.cede(np.array([80.0, 120.0, 200.0, 90.0]), np.zeros(4, dtype=np.int64)).tolist()
    [30.0, 50.0, 50.0, 20.0]
    """

    attachment: float
    limit: float
    reinstatements: int | None
    label: ClassVar[str] = "excess_of_loss"

    def __post_init__(self) -> None:
        _require_finite_non_negative("attachment", self.attachment)
        _require_positive("limit", self.limit)
        if self.reinstatements is not None and not (
            isinstance(self.reinstatements, int)
            and not isinstance(self.reinstatements, bool)
            and self.reinstatements >= 0
        ):
            raise CataggError(
                "reinstatements must be a non-negative integer or None, "
                f"got {self.reinstatements!r}."
            )

    @property
    def annual_limit(self) -> float:
        """Most the layer can pay in one year, ``limit * (1 + reinstatements)``."""
        if self.reinstatements is None:
            return math.inf
        return self.limit * (1 + self.reinstatements)

    def cede(self, retained: NDArray[np.float64], years: NDArray[np.int64]) -> NDArray[np.float64]:
        """Cede the layer loss on each occurrence, until the year's cover runs out.

        Parameters
        ----------
        retained
            Loss still retained on each occurrence before this structure.
        years
            Simulated year of each occurrence, sorted, chronological within year.

        Returns
        -------
        numpy.ndarray
            Amount ceded on each occurrence.
        """
        layer_loss = np.clip(retained - self.attachment, 0.0, self.limit)
        return cap_within_years(layer_loss, years, self.annual_limit)


@dataclass(frozen=True, kw_only=True)
class AggregateStopLoss:
    """Aggregate cover on each year's retained loss.

    The cover pays the part of the year's retained loss above ``attachment``,
    up to ``limit``. Attachment and limit are amounts of money, so this is also
    an aggregate excess of loss; a stop-loss quoted as a loss ratio converts to
    one given the subject premium.

    The annual recovery is allocated to occurrences in chronological order: the
    cover attaches part-way through the occurrence that takes the running total
    past ``attachment``, and stops paying on the one that exhausts ``limit``.
    That allocation is what makes a net OEP after an aggregate cover
    well-defined.

    Parameters
    ----------
    attachment
        Annual aggregate retention.
    limit
        Most the cover pays in a year; may be ``inf``.

    Examples
    --------
    Retained losses of 60, 70 and 90 build to 220. A 100 xs 100 cover attaches
    30 into the second occurrence and is exhausted within the third:

    >>> import numpy as np
    >>> cover = AggregateStopLoss(attachment=100.0, limit=100.0)
    >>> cover.cede(np.array([60.0, 70.0, 90.0]), np.zeros(3, dtype=np.int64)).tolist()
    [0.0, 30.0, 70.0]
    """

    attachment: float
    limit: float
    label: ClassVar[str] = "aggregate_stop_loss"

    def __post_init__(self) -> None:
        _require_finite_non_negative("attachment", self.attachment)
        _require_positive("limit", self.limit)

    def cede(self, retained: NDArray[np.float64], years: NDArray[np.int64]) -> NDArray[np.float64]:
        """Cede each occurrence's share of the year's aggregate recovery.

        Parameters
        ----------
        retained
            Loss still retained on each occurrence before this structure.
        years
            Simulated year of each occurrence, sorted, chronological within year.

        Returns
        -------
        numpy.ndarray
            Amount ceded on each occurrence.
        """
        deductible_left = np.maximum(self.attachment - cumulative_before(retained, years), 0.0)
        above_attachment = np.maximum(retained - deductible_left, 0.0)
        return cap_within_years(above_attachment, years, self.limit)
