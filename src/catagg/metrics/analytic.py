"""Closed-form quantities the simulation must reproduce.

These are the reference values for validation. The AAL in particular is the
single best sanity check in cat modelling: it is exact, it needs no simulation,
and a pipeline that gets it wrong is wrong everywhere else too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from catagg.exceptions import CataggError
from catagg.frequency.models import FrequencyModel, resolve_frequency
from catagg.io.elt import validate_elt

__all__ = ["aal_standard_error", "analytic_aal", "analytic_annual_loss_variance"]


def analytic_aal(elt: pd.DataFrame) -> float:
    """Return the average annual loss of an ELT, ``sum(rate * mean_loss)``.

    Independent of the frequency model: any count distribution with the
    specified mean gives the same AAL, which is why AAL alone cannot detect a
    frequency bug and `analytic_annual_loss_variance` is needed as well.

    Parameters
    ----------
    elt
        Event loss table.

    Returns
    -------
    float
        The exact AAL, in loss units.
    """
    elt = validate_elt(elt)
    return float(np.dot(elt["rate"].to_numpy(), elt["mean_loss"].to_numpy()))


def analytic_annual_loss_variance(
    elt: pd.DataFrame,
    frequency: str | FrequencyModel = "poisson",
    dispersion: float | None = None,
    *,
    group_by: str | None = "peril",
    secondary_uncertainty: bool = False,
) -> float:
    """Return the variance of the annual aggregate loss, in closed form.

    For each frequency group with excess-variance coefficient ``c``::

        Var = sum(rate * std_loss ** 2)
            + sum(rate * mean_loss ** 2)
            + c * sum(rate * mean_loss) ** 2

    The first term is present only when severity is stochastic. The middle term
    is the usual compound-Poisson contribution. The last is the clustering
    contribution, quadratic in the group's expected loss, which is what makes an
    over-dispersed frequency matter in the tail. Groups are assumed independent,
    which is what the engine currently does.

    Parameters
    ----------
    elt
        Event loss table.
    frequency
        Frequency model, as passed to `catagg.simulate`.
    dispersion
        Negative binomial shape parameter ``k``, if applicable.
    group_by
        Column defining the frequency groups, matching the simulation.
    secondary_uncertainty
        Include the ``sum(rate * std_loss ** 2)`` severity term. Leave ``False``
        to match a simulation run with degenerate severity.

    Returns
    -------
    float
        The variance of the annual aggregate loss.

    Raises
    ------
    CataggError
        If ``group_by`` is not a column of the ELT.
    """
    elt = validate_elt(elt)
    model = resolve_frequency(frequency, dispersion)
    coefficient = model.excess_variance_coefficient

    if group_by is None:
        labels: pd.Series = pd.Series(np.zeros(len(elt), dtype=np.int8), index=elt.index)
    elif group_by in elt.columns:
        labels = elt[group_by]
    else:
        raise CataggError(f"group_by={group_by!r} is not a column of the ELT.")

    rate = elt["rate"].to_numpy(dtype=np.float64)
    mean = elt["mean_loss"].to_numpy(dtype=np.float64)
    std = elt["std_loss"].to_numpy(dtype=np.float64)

    variance = float(np.dot(rate, mean**2))
    if secondary_uncertainty:
        variance += float(np.dot(rate, std**2))

    for label in labels.unique():
        mask = (labels == label).to_numpy()
        variance += coefficient * float(np.dot(rate[mask], mean[mask])) ** 2
    return variance


def aal_standard_error(
    elt: pd.DataFrame,
    n_years: int,
    frequency: str | FrequencyModel = "poisson",
    dispersion: float | None = None,
    *,
    group_by: str | None = "peril",
    secondary_uncertainty: bool = False,
) -> float:
    """Return the Monte Carlo standard error of an AAL from ``n_years`` years.

    ``sqrt(Var(annual loss) / n_years)``. This is the tolerance a convergence
    test should be written against: a fixed percentage tolerance either passes a
    broken engine on a fat-tailed portfolio or fails a correct one on a thin
    portfolio.

    Parameters
    ----------
    elt
        Event loss table.
    n_years
        Number of simulated years.
    frequency
        Frequency model used in the simulation.
    dispersion
        Negative binomial shape parameter ``k``, if applicable.
    group_by
        Column defining the frequency groups.
    secondary_uncertainty
        Whether the simulation used stochastic severity.

    Returns
    -------
    float
        The standard error of the simulated AAL, in loss units.

    Raises
    ------
    ValueError
        If ``n_years`` is not strictly positive.
    """
    if n_years <= 0:
        raise ValueError(f"n_years must be strictly positive, got {n_years}.")
    variance = analytic_annual_loss_variance(
        elt,
        frequency,
        dispersion,
        group_by=group_by,
        secondary_uncertainty=secondary_uncertainty,
    )
    return float(np.sqrt(variance / n_years))
