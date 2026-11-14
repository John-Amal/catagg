"""The year event loss table: schema and Parquet round-trip.

A YELT has one row per event *occurrence* per simulated year. Years in which
nothing happened have no rows, so the number of simulated years cannot be
recovered from the table itself and is carried alongside it in
``yelt.attrs["n_years"]``.

``occurrence_index`` is the chronological position of an occurrence within its
year, counting from zero. Anything that consumes cover event by event - a
per-occurrence layer with limited reinstatements, an aggregate cover allocated
back to occurrences - does so in this order, so code reading a YELT should sort
on ``(sim_year, occurrence_index)`` explicitly rather than rely on row order.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from catagg.exceptions import CataggError

__all__ = ["YELT_COLUMNS", "YELT_DTYPES", "n_years_of", "read_yelt", "write_yelt"]

YELT_DTYPES: Final[dict[str, str]] = {
    "sim_year": "int32",
    "event_id": "int64",
    "occurrence_index": "int32",
    "gross_loss": "float64",
}
"""Canonical YELT columns and their dtypes."""

YELT_COLUMNS: Final[tuple[str, ...]] = tuple(YELT_DTYPES)
"""Canonical YELT column names, in schema order."""

_N_YEARS_KEY: Final[bytes] = b"catagg.n_years"


def n_years_of(yelt: pd.DataFrame, n_years: int | None = None) -> int:
    """Resolve the simulated year count for a YELT.

    pandas drops ``DataFrame.attrs`` through some operations, so any function
    needing the year count takes an explicit override and falls back to the
    attribute. Using the wrong count silently biases every mean in the library,
    which is why this never guesses from ``sim_year.max()``.

    Parameters
    ----------
    yelt
        Year event loss table.
    n_years
        Explicit count. If given it wins, and is checked for consistency with
        the table's own ``sim_year`` range.

    Returns
    -------
    int
        The number of simulated years.

    Raises
    ------
    CataggError
        If no count is available, or the explicit count is too small for the
        years present in the table.
    """
    resolved = n_years if n_years is not None else yelt.attrs.get("n_years")
    if resolved is None:
        raise CataggError(
            "n_years is unknown for this YELT: it is not in `yelt.attrs` and was not "
            "passed explicitly. pandas drops `.attrs` through some operations - pass "
            "n_years= through, or reload the table with catagg.read_yelt."
        )
    resolved = int(resolved)
    if resolved <= 0:
        raise CataggError(f"n_years must be strictly positive, got {resolved}.")
    if len(yelt) and int(yelt["sim_year"].max()) >= resolved:
        raise CataggError(
            f"n_years={resolved} is inconsistent with the table, which contains "
            f"sim_year up to {int(yelt['sim_year'].max())}."
        )
    return resolved


def write_yelt(yelt: pd.DataFrame, path: str | Path, *, n_years: int | None = None) -> None:
    """Write a YELT to Parquet, preserving the simulated year count.

    The count is stored in the file's key-value metadata, so a round-trip
    restores it without a sidecar file.

    Parameters
    ----------
    yelt
        Year event loss table.
    path
        Destination path.
    n_years
        Explicit year count, if ``yelt.attrs`` no longer carries it.
    """
    resolved = n_years_of(yelt, n_years)
    table = pa.Table.from_pandas(yelt[list(YELT_COLUMNS)], preserve_index=False)
    metadata = dict(table.schema.metadata or {})
    metadata[_N_YEARS_KEY] = str(resolved).encode()
    pq.write_table(table.replace_schema_metadata(metadata), Path(path))


def read_yelt(path: str | Path) -> pd.DataFrame:
    """Read a YELT written by `write_yelt`, restoring ``attrs["n_years"]``.

    Parameters
    ----------
    path
        Source path.

    Returns
    -------
    pandas.DataFrame
        The year event loss table.

    Raises
    ------
    CataggError
        If the file carries no year count, meaning it was not written by
        `write_yelt` and its empty years cannot be accounted for.
    """
    table = pq.read_table(Path(path))
    metadata = table.schema.metadata or {}
    if _N_YEARS_KEY not in metadata:
        raise CataggError(
            f"{path} carries no catagg.n_years metadata; it was not written by catagg.write_yelt."
        )
    frame: pd.DataFrame = table.to_pandas()
    frame = frame.astype(YELT_DTYPES)
    frame.attrs["n_years"] = int(metadata[_N_YEARS_KEY])
    return frame
