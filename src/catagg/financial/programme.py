"""A reinsurance programme: structures applied in order to a YELT."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from catagg.exceptions import CataggError
from catagg.financial.structures import Structure
from catagg.io.yelt import n_years_of

__all__ = ["Programme", "ProgrammeResult"]

_REQUIRED_COLUMNS = ("sim_year", "event_id", "occurrence_index", "gross_loss")
_RESULT_COLUMNS = (*_REQUIRED_COLUMNS, "ceded_loss", "net_loss")


@dataclass(frozen=True)
class ProgrammeResult:
    """Gross, ceded and net loss on every simulated occurrence.

    Attributes
    ----------
    yelt
        A YELT with columns ``sim_year``, ``event_id``, ``occurrence_index``,
        ``gross_loss``, ``ceded_loss`` and ``net_loss``, sorted by year and then
        chronologically, with ``attrs["n_years"]`` set. Pass it to `ep_curve` or
        `to_ylt` with ``column="net_loss"`` or ``column="ceded_loss"``.
    ceded_by_structure
        Amount ceded to each structure on each occurrence, aligned with
        ``yelt``. Columns are labelled ``"<position>_<label>"`` in programme
        order, counting from 1.
    """

    yelt: pd.DataFrame
    ceded_by_structure: pd.DataFrame

    @property
    def n_years(self) -> int:
        """Number of simulated years, including years with no occurrences."""
        return n_years_of(self.yelt)


class Programme:
    """Reinsurance structures applied in order.

    Each structure protects what the structures before it left retained, so a
    quota share listed first cedes a share of the gross loss and an excess of
    loss listed after it sees only the retained share. Order therefore matters,
    and it is the order given here.

    Parameters
    ----------
    structures
        The structures, in the order they apply. Any object satisfying the
        `Structure` protocol is accepted.

    Examples
    --------
    >>> import catagg as ca
    >>> elt = ca.swiss_multiperil_elt(n_events=200, seed=0)
    >>> yelt = ca.simulate(elt, n_years=1_000, seed=1)
    >>> programme = ca.Programme(
    ...     [
    ...         ca.QuotaShare(cession=0.2),
    ...         ca.ExcessOfLoss(attachment=50e6, limit=50e6, reinstatements=2),
    ...         ca.AggregateStopLoss(attachment=100e6, limit=100e6),
    ...     ]
    ... )
    >>> result = programme.apply(yelt)
    >>> list(result.yelt.columns[-2:])
    ['ceded_loss', 'net_loss']
    >>> net = ca.ep_curve(result.yelt, kind="OEP", column="net_loss")
    >>> net.n_years
    1000
    """

    def __init__(self, structures: Sequence[Structure]) -> None:
        for structure in structures:
            if not isinstance(structure, Structure):
                raise CataggError(
                    f"{structure!r} is not a reinsurance structure: it needs a "
                    "label and a cede(retained, years) method."
                )
        self._structures = tuple(structures)

    @property
    def structures(self) -> tuple[Structure, ...]:
        """The structures, in the order they apply."""
        return self._structures

    def __repr__(self) -> str:
        return f"Programme({list(self._structures)!r})"

    def apply(self, yelt: pd.DataFrame) -> ProgrammeResult:
        """Apply the programme to every simulated occurrence.

        Parameters
        ----------
        yelt
            Year event loss table, as returned by `catagg.simulate`. Rows are
            put in order of ``(sim_year, occurrence_index)`` before anything is
            ceded; the order they arrive in is not relied on.

        Returns
        -------
        ProgrammeResult
            Gross, ceded and net loss per occurrence, and each structure's
            cession.

        Raises
        ------
        CataggError
            If a required column is missing, two rows share a ``sim_year`` and
            ``occurrence_index``, the year count is unknown, or a structure
            cedes a negative amount or more than it was offered.
        """
        missing = [column for column in _REQUIRED_COLUMNS if column not in yelt.columns]
        if missing:
            raise CataggError(f"the YELT is missing required columns: {missing}.")
        n_years = n_years_of(yelt)

        ordered = yelt.loc[:, list(_REQUIRED_COLUMNS)].sort_values(
            ["sim_year", "occurrence_index"], kind="stable", ignore_index=True
        )
        years = ordered["sim_year"].to_numpy(dtype=np.int64)
        index = ordered["occurrence_index"].to_numpy(dtype=np.int64)
        if years.shape[0] > 1 and np.any((years[1:] == years[:-1]) & (index[1:] == index[:-1])):
            raise CataggError(
                "two occurrences share a sim_year and occurrence_index, so their "
                "order within the year is undefined."
            )

        gross = ordered["gross_loss"].to_numpy(dtype=np.float64)
        retained = gross.copy()
        ceded_columns: dict[str, np.ndarray] = {}
        for position, structure in enumerate(self._structures, start=1):
            ceded = np.asarray(structure.cede(retained, years), dtype=np.float64)
            if ceded.shape != retained.shape or np.any(ceded < 0.0) or np.any(ceded > retained):
                raise CataggError(
                    f"structure {position} ({structure.label}) ceded a negative "
                    "amount, or more than the loss it was offered."
                )
            ceded_columns[f"{position}_{structure.label}"] = ceded
            retained = retained - ceded

        # One subtraction, so net + ceded recovers gross to within one unit in
        # the last place on every occurrence - the float form of an identity.
        result = ordered.assign(ceded_loss=gross - retained, net_loss=retained)
        result = result.loc[:, list(_RESULT_COLUMNS)]
        result.attrs["n_years"] = n_years
        by_structure = pd.DataFrame(ceded_columns, index=result.index)
        return ProgrammeResult(yelt=result, ceded_by_structure=by_structure)
