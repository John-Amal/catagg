"""The year loop: ELT in, year event loss table out.

Occurrences are generated in two stages per event group.

1. Draw the group's annual count ``N`` from the frequency model.
2. Allocate each of those ``N`` occurrences to an event, independently, with
   probability proportional to the event's rate.

For the Poisson case this is exactly equivalent to drawing each event's count
from an independent ``Poisson(rate_i)``: Poisson thinning is exact, not an
approximation. For the negative binomial it keeps every marginal rate exact,
since ``E[N_i] = E[N] * p_i = rate_i``, while making the group total
over-dispersed, because the shared draw of ``N`` induces a positive covariance
``rate_i * rate_j / k`` between any two events in the group. That covariance is
the clustering; per-event negative binomial draws would not produce it.
"""

from __future__ import annotations

from itertools import pairwise
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from catagg.exceptions import CataggError
from catagg.frequency.models import FrequencyModel, resolve_frequency
from catagg.io.elt import validate_elt
from catagg.io.yelt import YELT_DTYPES, n_years_of

__all__ = ["simulate", "to_ylt"]

_DEFAULT_CHUNK_YEARS: Final[int] = 25_000


def _group_positions(
    elt: pd.DataFrame, group_by: str | None
) -> list[tuple[object, NDArray[np.int64]]]:
    """Split the ELT into frequency groups, ordered by label for reproducibility.

    Parameters
    ----------
    elt
        Validated event loss table.
    group_by
        Column defining the groups, or ``None`` for a single group.

    Returns
    -------
    list of tuple
        Pairs of group label and the integer row positions belonging to it.

    Raises
    ------
    CataggError
        If ``group_by`` is not a column of the ELT.
    """
    if group_by is None:
        return [(None, np.arange(len(elt), dtype=np.int64))]
    if group_by not in elt.columns:
        raise CataggError(
            f"group_by={group_by!r} is not a column of the ELT; columns are "
            f"{', '.join(map(str, elt.columns))}."
        )
    labels = elt[group_by]
    return [
        (label, np.flatnonzero((labels == label).to_numpy()).astype(np.int64))
        for label in sorted(labels.unique())
    ]


def _allocate_occurrences(
    rates: NDArray[np.float64],
    counts: NDArray[np.int64],
    rng: np.random.Generator,
    chunk_years: int,
) -> NDArray[np.int64]:
    """Assign each occurrence to a within-group event index, in year order.

    Occurrences are drawn by inverse-CDF sampling on the rate-proportional
    distribution. Uniforms are drawn in chunks purely to bound transient memory;
    because the counts for every year are drawn before any uniform is, the stream
    of uniforms is identical for any ``chunk_years``, so results do not depend on
    the chunk size.

    Parameters
    ----------
    rates
        Event rates within the group.
    counts
        Occurrences per year, shape ``(n_years,)``.
    rng
        Source of randomness.
    chunk_years
        Years per uniform-sampling batch.

    Returns
    -------
    numpy.ndarray
        Index into ``rates`` for each occurrence, in year order.
    """
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)

    cumulative = np.cumsum(rates)
    cumulative /= cumulative[-1]
    cumulative[-1] = 1.0

    n_years = counts.shape[0]
    boundaries = np.concatenate(
        [np.arange(0, n_years, chunk_years, dtype=np.int64), np.array([n_years], dtype=np.int64)]
    )
    pieces: list[NDArray[np.int64]] = []
    for start, stop in pairwise(boundaries):
        draws = int(counts[start:stop].sum())
        if draws == 0:
            continue
        uniforms = rng.random(draws)
        pieces.append(np.searchsorted(cumulative, uniforms, side="right"))

    allocated: NDArray[np.int64] = np.concatenate(pieces).astype(np.int64, copy=False)
    np.minimum(allocated, rates.shape[0] - 1, out=allocated)
    return allocated


def simulate(
    elt: pd.DataFrame,
    n_years: int,
    *,
    frequency: str | FrequencyModel = "poisson",
    dispersion: float | None = None,
    secondary_uncertainty: bool = False,
    group_by: str | None = "peril",
    seed: int | None = None,
    chunk_years: int = _DEFAULT_CHUNK_YEARS,
) -> pd.DataFrame:
    """Simulate a year event loss table from an event loss table.

    Parameters
    ----------
    elt
        Event loss table. Validated on entry; see `catagg.validate_elt`.
    n_years
        Number of years to simulate.
    frequency
        ``"poisson"``, ``"negative_binomial"``, or a
        `catagg.frequency.FrequencyModel` instance.
    dispersion
        Negative binomial shape parameter ``k``. Required for
        ``frequency="negative_binomial"``.
    secondary_uncertainty
        Not yet implemented. With ``False``, loss given occurrence is degenerate
        at ``mean_loss``, which leaves the AAL exact and every tail metric
        understated.
    group_by
        ELT column defining the frequency groups. The frequency model is applied
        to each group's annual total. ``None`` treats the whole ELT as one group.
    seed
        Seed for the generator. ``None`` draws from OS entropy.
    chunk_years
        Years per uniform-sampling batch. Affects peak memory only; results are
        identical for any value.

    Returns
    -------
    pandas.DataFrame
        A YELT with columns ``sim_year``, ``event_id``, ``occurrence_index`` and
        ``gross_loss``, sorted by ``sim_year`` then ``occurrence_index``, with
        ``attrs["n_years"]`` set. Years with no occurrences contribute no rows.

    Raises
    ------
    NotImplementedError
        If ``secondary_uncertainty`` is ``True``.
    CataggError
        If ``group_by`` is not an ELT column.
    ValueError
        If ``n_years`` or ``chunk_years`` is not positive, or the frequency
        arguments are inconsistent.

    Examples
    --------
    >>> import catagg as ca
    >>> elt = ca.swiss_multiperil_elt(n_events=100, seed=0)
    >>> yelt = ca.simulate(elt, n_years=1_000, seed=0)
    >>> list(yelt.columns)
    ['sim_year', 'event_id', 'occurrence_index', 'gross_loss']
    """
    if secondary_uncertainty:
        raise NotImplementedError(
            "secondary_uncertainty is not implemented in this version. Loss given "
            "occurrence is currently degenerate at mean_loss; the beta-with-point-mass "
            "sampler lands in catagg.severity."
        )
    if n_years <= 0:
        raise ValueError(f"n_years must be strictly positive, got {n_years}.")
    if chunk_years <= 0:
        raise ValueError(f"chunk_years must be strictly positive, got {chunk_years}.")

    elt = validate_elt(elt)
    model = resolve_frequency(frequency, dispersion)
    groups = _group_positions(elt, group_by)

    rate = elt["rate"].to_numpy(dtype=np.float64)
    event_id = elt["event_id"].to_numpy(dtype=np.int64)
    mean_loss = elt["mean_loss"].to_numpy(dtype=np.float64)

    seeds = np.random.SeedSequence(seed).spawn(len(groups))
    total_counts = np.zeros(n_years, dtype=np.int64)
    year_pieces: list[NDArray[np.int32]] = []
    position_pieces: list[NDArray[np.int64]] = []

    for (_, positions), child in zip(groups, seeds, strict=True):
        rng = np.random.default_rng(child)
        group_rates = rate[positions]
        counts = model.sample_annual_counts(float(group_rates.sum()), n_years, rng)
        total_counts += counts

        within_group = _allocate_occurrences(group_rates, counts, rng, chunk_years)
        position_pieces.append(positions[within_group])
        year_pieces.append(np.repeat(np.arange(n_years, dtype=np.int32), counts))

    occurrence_years = np.concatenate(year_pieces) if year_pieces else np.empty(0, dtype=np.int32)
    occurrence_positions = (
        np.concatenate(position_pieces) if position_pieces else np.empty(0, dtype=np.int64)
    )

    order = np.argsort(occurrence_years, kind="stable")
    occurrence_positions = occurrence_positions[order]

    n_occurrences = occurrence_positions.shape[0]
    starts = np.cumsum(total_counts) - total_counts
    occurrence_index = (
        np.arange(n_occurrences, dtype=np.int64) - np.repeat(starts, total_counts)
    ).astype(np.int32, copy=False)

    yelt = pd.DataFrame(
        {
            "sim_year": np.repeat(np.arange(n_years, dtype=np.int32), total_counts),
            "event_id": event_id[occurrence_positions],
            "occurrence_index": occurrence_index,
            "gross_loss": mean_loss[occurrence_positions],
        }
    ).astype(YELT_DTYPES)
    yelt.attrs["n_years"] = int(n_years)
    return yelt


def to_ylt(yelt: pd.DataFrame, n_years: int | None = None) -> pd.DataFrame:
    """Collapse a YELT to one row per simulated year.

    Every year in ``range(n_years)`` appears, including the ones in which nothing
    happened. Those years carry zero loss and are not optional: dropping them, as
    a bare ``groupby("sim_year")`` silently does, divides by the number of
    *non-empty* years and inflates the AAL, by a factor of roughly
    ``1 / (1 - exp(-Lambda))`` for a Poisson portfolio.

    Parameters
    ----------
    yelt
        Year event loss table.
    n_years
        Explicit year count if ``yelt.attrs`` no longer carries it.

    Returns
    -------
    pandas.DataFrame
        A YLT with columns ``sim_year``, ``aggregate_loss`` and
        ``max_occurrence_loss``.
    """
    resolved = n_years_of(yelt, n_years)
    years = np.arange(resolved, dtype=np.int32)
    sim_year = yelt["sim_year"].to_numpy(dtype=np.int64)
    gross_loss = yelt["gross_loss"].to_numpy(dtype=np.float64)

    aggregate = np.bincount(sim_year, weights=gross_loss, minlength=resolved)
    maximum = np.zeros(resolved, dtype=np.float64)
    np.maximum.at(maximum, sim_year, gross_loss)

    ylt = pd.DataFrame(
        {"sim_year": years, "aggregate_loss": aggregate, "max_occurrence_loss": maximum}
    )
    ylt.attrs["n_years"] = resolved
    return ylt
