"""Synthetic example portfolios.

Every example, test and figure in this project runs from a generator rather than
a data file, so a clean clone reproduces everything with no downloads and no
licensing questions.

!!! warning "Illustrative parameters"

    The parameters below are **not calibrated to any real portfolio, market or
    hazard model**. They were chosen to give a three-peril structure with a
    plausible frequency-severity shape and nothing more. Do not read any of the
    numbers as an estimate of Swiss hail, flood or windstorm risk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

__all__ = ["SWISS_PERILS", "PerilSpec", "swiss_multiperil_elt"]


@dataclass(frozen=True)
class PerilSpec:
    """Generator parameters for one synthetic peril.

    Attributes
    ----------
    name
        Peril label.
    annual_rate
        Expected number of events per year for the peril as a whole.
    share_of_events
        Share of the event set allocated to this peril.
    log_mean
        Mean of the log of the event mean loss.
    log_sigma
        Standard deviation of the log of the event mean loss.
    rate_severity_exponent
        Rates are set proportional to ``mean_loss ** -exponent`` before
        rescaling, so that rarer events are larger. Zero makes rate and severity
        independent.
    coefficient_of_variation
        Ratio of ``std_loss`` to ``mean_loss``.
    exposure_multiple
        ``exposure_value`` as a multiple of ``mean_loss``.
    """

    name: str
    annual_rate: float
    share_of_events: float
    log_mean: float
    log_sigma: float
    rate_severity_exponent: float
    coefficient_of_variation: float
    exposure_multiple: float


SWISS_PERILS: Final[tuple[PerilSpec, ...]] = (
    PerilSpec("hail", 6.0, 0.50, np.log(2.0e6), 0.9, 0.6, 1.0, 5.0),
    PerilSpec("flood", 2.5, 0.30, np.log(6.0e6), 1.1, 0.6, 1.1, 4.0),
    PerilSpec("windstorm", 0.8, 0.20, np.log(2.0e7), 1.3, 0.6, 1.2, 4.0),
)
"""Illustrative three-peril structure. See the module warning."""

_REGIONS: Final[tuple[str, ...]] = ("Plateau", "Alpine", "Jura")

# Safety factor keeping the generated std_loss strictly inside the feasible
# variance bound, so the generator can never emit an ELT its own validator rejects.
_VARIANCE_SAFETY: Final[float] = 0.99


def swiss_multiperil_elt(
    n_events: int = 1_000,
    seed: int | None = 0,
    perils: tuple[PerilSpec, ...] = SWISS_PERILS,
) -> pd.DataFrame:
    """Generate a synthetic multi-peril event loss table.

    Event mean losses are lognormal; rates are proportional to a negative power
    of the mean loss and then rescaled so each peril's rates sum exactly to its
    ``annual_rate``, which makes the analytic AAL of the result a clean
    reference value. ``std_loss`` is a fixed multiple of ``mean_loss``, clipped
    to stay strictly inside the bound enforced by `catagg.validate_elt`.

    Parameters
    ----------
    n_events
        Total number of events across all perils.
    seed
        Seed for reproducibility. ``None`` draws from OS entropy.
    perils
        Peril specifications.

    Returns
    -------
    pandas.DataFrame
        A valid ELT with the canonical columns.

    Raises
    ------
    ValueError
        If ``n_events`` is not strictly positive.
    """
    if n_events <= 0:
        raise ValueError(f"n_events must be strictly positive, got {n_events}.")

    rng = np.random.default_rng(seed)
    frames: list[pd.DataFrame] = []
    next_event_id = 1

    for spec in perils:
        size = max(1, round(n_events * spec.share_of_events))
        mean_loss = rng.lognormal(mean=spec.log_mean, sigma=spec.log_sigma, size=size)

        weights = mean_loss**-spec.rate_severity_exponent
        rate = spec.annual_rate * weights / weights.sum()

        exposure_value = mean_loss * spec.exposure_multiple
        max_std = np.sqrt(mean_loss * (exposure_value - mean_loss))
        std_loss = np.minimum(spec.coefficient_of_variation * mean_loss, _VARIANCE_SAFETY * max_std)

        frames.append(
            pd.DataFrame(
                {
                    "event_id": np.arange(next_event_id, next_event_id + size, dtype=np.int64),
                    "rate": rate,
                    "mean_loss": mean_loss,
                    "std_loss": std_loss,
                    "exposure_value": exposure_value,
                    "peril": spec.name,
                    "region": rng.choice(np.array(_REGIONS), size=size),
                }
            )
        )
        next_event_id += size

    from catagg.io.elt import validate_elt

    return validate_elt(pd.concat(frames, ignore_index=True))
