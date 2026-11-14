"""Running totals within each simulated year.

Every limited cover in a programme is a running total that resets each year:
reinstatements are consumed occurrence by occurrence, and an aggregate cover
attaches once the year's retained loss has built past its deductible. These
helpers implement both without a Python loop over years.

Arrays are aligned occurrence by occurrence and must be sorted by year, then
chronologically within the year; `Programme.apply` guarantees that.

Two numerical choices matter here and are deliberate:

* Running totals are accumulated **within each year** (a grouped cumulative
  sum), never as one cumulative sum over the whole table minus each year's
  starting value. The latter subtracts two totals of order ``n_years`` times
  the annual loss and loses most of its significant digits to cancellation.
* A capped cession is written as ``min(x, remaining capacity)``, never as the
  difference of two capped running totals. The difference form does not return
  ``x`` exactly when the cap is slack, which would let a cession exceed the loss
  it came from by one rounding error.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

__all__ = ["cap_within_years", "cumulative_before"]


def cumulative_before(values: NDArray[np.float64], years: NDArray[np.int64]) -> NDArray[np.float64]:
    """Sum the values of earlier occurrences in the same year.

    Parameters
    ----------
    values
        One value per occurrence.
    years
        Simulated year of each occurrence, sorted ascending.

    Returns
    -------
    numpy.ndarray
        For each occurrence, the sum of ``values`` over the occurrences before it
        in its own year; zero for the first occurrence of every year.
    """
    if values.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    inclusive = pd.Series(values).groupby(years, sort=False).cumsum().to_numpy(dtype=np.float64)
    before: NDArray[np.float64] = np.empty(inclusive.shape[0], dtype=np.float64)
    before[0] = 0.0
    before[1:] = inclusive[:-1]
    first_of_year = np.empty(years.shape[0], dtype=np.bool_)
    first_of_year[0] = True
    first_of_year[1:] = years[1:] != years[:-1]
    before[first_of_year] = 0.0
    return before


def cap_within_years(
    values: NDArray[np.float64], years: NDArray[np.int64], cap: float
) -> NDArray[np.float64]:
    """Pay each value in order until the year's total reaches ``cap``.

    Parameters
    ----------
    values
        Non-negative amount claimed by each occurrence.
    years
        Simulated year of each occurrence, sorted ascending.
    cap
        Most that can be paid in any one year; ``inf`` for no cap.

    Returns
    -------
    numpy.ndarray
        Amount paid on each occurrence: its full value while capacity lasts, the
        remainder on the occurrence that exhausts it, and zero after that.

    Examples
    --------
    >>> import numpy as np
    >>> paid = cap_within_years(
    ...     np.array([30.0, 50.0, 50.0, 40.0]), np.zeros(4, dtype=np.int64), cap=150.0
    ... )
    >>> paid.tolist()
    [30.0, 50.0, 50.0, 20.0]
    """
    if np.isinf(cap):
        return values.copy()
    remaining = np.maximum(cap - cumulative_before(values, years), 0.0)
    paid: NDArray[np.float64] = np.minimum(values, remaining)
    return paid
