"""The event loss table: schema, validation, and readers.

An ELT has one row per modelled event. It is the input to the whole engine, so
every downstream guarantee rests on the checks in `validate_elt`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from pandas.api.types import pandas_dtype

from catagg.exceptions import ELTValidationError

__all__ = ["ELT_COLUMNS", "ELT_DTYPES", "read_elt", "validate_elt"]

ELT_DTYPES: Final[dict[str, str]] = {
    "event_id": "int64",
    "rate": "float64",
    "mean_loss": "float64",
    "std_loss": "float64",
    "exposure_value": "float64",
    "peril": "string",
    "region": "string",
}
"""Canonical ELT columns and their dtypes after normalisation."""

ELT_COLUMNS: Final[tuple[str, ...]] = tuple(ELT_DTYPES)
"""Canonical ELT column names, in schema order."""

# Relative slack when checking the variance bound, so that an ELT written by
# another tool in float32 is not rejected for last-bit rounding.
_VARIANCE_RTOL: Final[float] = 1e-9

# Number of offending event ids quoted in an error message.
_MAX_REPORTED: Final[int] = 5


def _report(event_ids: pd.Series, mask: NDArray[np.bool_]) -> str:
    """Format up to five offending event ids for an error message.

    Parameters
    ----------
    event_ids
        The ``event_id`` column, aligned with ``mask``.
    mask
        Boolean mask selecting the offending rows. Always a numpy array, so
        that every check in `validate_elt` builds its mask the same way.

    Returns
    -------
    str
        A comma-separated list of ids, truncated with a count.
    """
    offenders = event_ids.to_numpy()[mask]
    shown = ", ".join(str(value) for value in offenders[:_MAX_REPORTED])
    if len(offenders) > _MAX_REPORTED:
        shown += f", ... ({len(offenders)} rows in total)"
    return shown


def validate_elt(elt: pd.DataFrame) -> pd.DataFrame:
    """Validate an event loss table and return a normalised copy.

    Normalisation means canonical columns coerced to ``ELT_DTYPES`` and moved to
    the front in schema order, any extra columns preserved after them, and the
    index reset.

    The checks are: all canonical columns present; ``event_id`` integral,
    non-null and unique; ``rate`` finite and strictly positive; ``mean_loss`` and
    ``std_loss`` finite and non-negative; ``exposure_value`` finite and strictly
    positive; ``mean_loss`` at most ``exposure_value``; the variance bound below;
    and ``peril`` and ``region`` present and non-empty.

    The variance bound is the one that catches real ELTs. For any loss
    distribution supported on ``[0, exposure_value]`` with mean ``mean_loss``,
    the variance is maximised by the two-point distribution placing all its mass
    at the endpoints, giving ``mean_loss * (exposure_value - mean_loss)``. A row
    exceeding that describes a distribution which cannot exist, so no severity
    model can honour it. Catching it here rather than inside the sampler is what
    stops the secondary uncertainty module from silently clipping the moments it
    was asked to reproduce.

    Parameters
    ----------
    elt
        Candidate event loss table.

    Returns
    -------
    pandas.DataFrame
        A validated, normalised copy. The input is never modified.

    Raises
    ------
    ELTValidationError
        If any check fails. The message lists every failure, not just the first.

    Examples
    --------
    >>> import pandas as pd
    >>> from catagg import validate_elt
    >>> elt = pd.DataFrame(
    ...     {
    ...         "event_id": [1],
    ...         "rate": [0.5],
    ...         "mean_loss": [1.0e6],
    ...         "std_loss": [5.0e5],
    ...         "exposure_value": [5.0e6],
    ...         "peril": ["hail"],
    ...         "region": ["Plateau"],
    ...     }
    ... )
    >>> list(validate_elt(elt).columns)
    ['event_id', 'rate', 'mean_loss', 'std_loss', 'exposure_value', 'peril', 'region']
    """
    if not isinstance(elt, pd.DataFrame):  # pragma: no cover - defensive
        raise ELTValidationError(f"ELT must be a pandas DataFrame, got {type(elt).__name__}.")

    missing = [column for column in ELT_COLUMNS if column not in elt.columns]
    if missing:
        raise ELTValidationError(
            f"ELT is missing required column(s): {', '.join(missing)}. "
            f"Required schema is {', '.join(ELT_COLUMNS)}."
        )

    frame = elt.copy()
    problems: list[str] = []

    for column, dtype in ELT_DTYPES.items():
        try:
            frame[column] = frame[column].astype(pandas_dtype(dtype))
        except (TypeError, ValueError) as error:
            problems.append(f"column {column!r} cannot be cast to {dtype}: {error}")

    if problems:
        raise ELTValidationError("ELT failed validation:\n  - " + "\n  - ".join(problems))

    extra = [column for column in frame.columns if column not in ELT_COLUMNS]
    frame = frame[[*ELT_COLUMNS, *extra]].reset_index(drop=True)

    event_id = frame["event_id"]
    if event_id.isna().any():
        problems.append("event_id contains nulls")
    duplicated = event_id.duplicated(keep=False).to_numpy(dtype=bool)
    if bool(duplicated.any()):
        problems.append(f"event_id must be unique; duplicated: {_report(event_id, duplicated)}")

    for column in ("rate", "exposure_value"):
        values = frame[column].to_numpy(dtype=np.float64)
        bad = ~np.isfinite(values) | (values <= 0.0)
        if bool(bad.any()):
            problems.append(
                f"{column} must be finite and strictly positive: {_report(event_id, bad)}"
            )

    for column in ("mean_loss", "std_loss"):
        values = frame[column].to_numpy(dtype=np.float64)
        bad = ~np.isfinite(values) | (values < 0.0)
        if bool(bad.any()):
            problems.append(f"{column} must be finite and non-negative: {_report(event_id, bad)}")

    if not problems:
        mean = frame["mean_loss"].to_numpy()
        std = frame["std_loss"].to_numpy()
        exposure = frame["exposure_value"].to_numpy()

        over_exposed = mean > exposure
        if bool(over_exposed.any()):
            problems.append(
                f"mean_loss must not exceed exposure_value: {_report(event_id, over_exposed)}"
            )

        bound = mean * (exposure - mean)
        infeasible = (std**2 > bound * (1.0 + _VARIANCE_RTOL)) & ~over_exposed
        if bool(infeasible.any()):
            problems.append(
                "std_loss exceeds the largest standard deviation a distribution on "
                "[0, exposure_value] with this mean can have, "
                "sqrt(mean_loss * (exposure_value - mean_loss)): "
                f"{_report(event_id, infeasible)}"
            )

    for column in ("peril", "region"):
        labels = frame[column]
        bad = (labels.isna() | (labels.fillna("").str.strip() == "")).to_numpy(dtype=bool)
        if bool(bad.any()):
            problems.append(f"{column} must be a non-empty label: {_report(event_id, bad)}")

    if problems:
        raise ELTValidationError("ELT failed validation:\n  - " + "\n  - ".join(problems))

    return frame


def read_elt(path: str | Path) -> pd.DataFrame:
    """Read an event loss table from Parquet or CSV and validate it.

    The format is chosen from the file suffix: ``.parquet`` and ``.pq`` go
    through pyarrow, ``.csv`` through `pandas.read_csv`.

    Parameters
    ----------
    path
        Path to the file.

    Returns
    -------
    pandas.DataFrame
        A validated, normalised ELT.

    Raises
    ------
    ELTValidationError
        If the table does not satisfy the schema.
    ValueError
        If the suffix is not a recognised format.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    elif suffix == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported ELT format {suffix!r}; expected .parquet or .csv.")
    return validate_elt(frame)
