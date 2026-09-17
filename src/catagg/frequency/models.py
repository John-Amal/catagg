"""Annual event-count models.

Both models are defined on the *annual count for a group of events*, not on each
event separately. This is deliberate, and it is the whole point of offering a
negative binomial at all.

If a negative binomial with dispersion ``k`` is applied to each event
independently, that event's count has variance ``rate + rate ** 2 / k``. Event
rates in a real ELT are of order ``1e-3``, so the quadratic term is a millionth
of the linear one and the portfolio behaves exactly as if it were Poisson: the
over-dispersion has been specified and then thrown away.

Applying the model to the group total and allocating occurrences to events in
proportion to their rates keeps every marginal rate exact while making the
*aggregate* count over-dispersed, which is what clustering means. See
`catagg.simulate.engine` for the allocation step and for why the Poisson case
reduces exactly to independent per-event Poisson draws.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

__all__ = ["FrequencyModel", "NegativeBinomial", "Poisson", "resolve_frequency"]


@dataclass(frozen=True)
class FrequencyModel:
    """Base class for annual event-count models.

    For validation purposes a model is fully described by
    `excess_variance_coefficient`: the constant ``c`` for which the annual count
    of a group with total rate ``Lambda`` has variance
    ``Lambda + c * Lambda ** 2``.
    """

    @property
    def excess_variance_coefficient(self) -> float:
        """Return the ``c`` in ``Var(N) = Lambda + c * Lambda ** 2``.

        Zero means Poisson.
        """
        raise NotImplementedError

    def annual_count_variance(self, group_rate: float) -> float:
        """Return the analytic variance of a group's annual event count.

        Parameters
        ----------
        group_rate
            Total annual rate of the group, ``sum(rate)``.

        Returns
        -------
        float
            The variance of the annual count.
        """
        return float(group_rate + self.excess_variance_coefficient * group_rate**2)

    def sample_annual_counts(
        self, group_rate: float, n_years: int, rng: Generator
    ) -> NDArray[np.int64]:
        """Draw the number of occurrences in each of ``n_years`` years.

        Parameters
        ----------
        group_rate
            Total annual rate of the group, ``sum(rate)``.
        n_years
            Number of years to draw.
        rng
            Source of randomness.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_years,)``, non-negative integer counts.
        """
        raise NotImplementedError


@dataclass(frozen=True)
class Poisson(FrequencyModel):
    """Poisson annual counts: mean and variance both equal the group rate."""

    @property
    def excess_variance_coefficient(self) -> float:
        """Return zero, always."""
        return 0.0

    def sample_annual_counts(
        self, group_rate: float, n_years: int, rng: Generator
    ) -> NDArray[np.int64]:
        """Draw Poisson counts with mean ``group_rate``.

        Parameters
        ----------
        group_rate
            Total annual rate of the group.
        n_years
            Number of years to draw.
        rng
            Source of randomness.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_years,)``, non-negative integer counts.
        """
        return rng.poisson(lam=group_rate, size=n_years).astype(np.int64, copy=False)


@dataclass(frozen=True)
class NegativeBinomial(FrequencyModel):
    """Negative binomial annual counts, over-dispersed relative to Poisson.

    Parameterised by the shape, or contagion, parameter ``dispersion`` = ``k``,
    so the annual count of a group with total rate ``Lambda`` has mean
    ``Lambda`` and variance ``Lambda + Lambda ** 2 / k``. Small ``k`` means heavy
    clustering; ``k`` to infinity recovers the Poisson.

    To specify a variance-to-mean ratio instead, use
    `NegativeBinomial.from_variance_to_mean_ratio`.

    Attributes
    ----------
    dispersion
        Shape parameter ``k``, strictly positive.
    """

    dispersion: float

    def __post_init__(self) -> None:
        """Reject a non-positive or non-finite dispersion."""
        if not np.isfinite(self.dispersion) or self.dispersion <= 0.0:
            raise ValueError(
                f"NegativeBinomial dispersion must be finite and strictly positive, "
                f"got {self.dispersion!r}."
            )

    @classmethod
    def from_variance_to_mean_ratio(cls, ratio: float, group_rate: float) -> NegativeBinomial:
        """Build the model whose annual count variance is ``ratio`` times its mean.

        The mapping depends on the group rate, because a variance-to-mean ratio
        is not scale-free for the negative binomial: ``k = Lambda / (phi - 1)``.

        Parameters
        ----------
        ratio
            Variance-to-mean ratio ``phi``, strictly greater than one.
        group_rate
            Total annual rate ``Lambda`` of the group.

        Returns
        -------
        NegativeBinomial
            The corresponding model.

        Raises
        ------
        ValueError
            If ``ratio`` is not strictly greater than one.
        """
        if ratio <= 1.0:
            raise ValueError(
                f"variance-to-mean ratio must exceed 1 for a negative binomial, got {ratio!r}; "
                "use Poisson() for a ratio of exactly 1."
            )
        return cls(dispersion=group_rate / (ratio - 1.0))

    @property
    def excess_variance_coefficient(self) -> float:
        """Return the reciprocal of the dispersion."""
        return 1.0 / self.dispersion

    def sample_annual_counts(
        self, group_rate: float, n_years: int, rng: Generator
    ) -> NDArray[np.int64]:
        """Draw negative binomial counts with mean ``group_rate``.

        Parameters
        ----------
        group_rate
            Total annual rate of the group.
        n_years
            Number of years to draw.
        rng
            Source of randomness.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_years,)``, non-negative integer counts.
        """
        success_probability = self.dispersion / (self.dispersion + group_rate)
        return rng.negative_binomial(n=self.dispersion, p=success_probability, size=n_years).astype(
            np.int64, copy=False
        )


def resolve_frequency(
    frequency: str | FrequencyModel, dispersion: float | None = None
) -> FrequencyModel:
    """Turn a ``frequency=`` argument into a `FrequencyModel`.

    Parameters
    ----------
    frequency
        ``"poisson"``, ``"negative_binomial"`` or ``"nb"``, or a model instance,
        which is returned unchanged.
    dispersion
        Required for ``"negative_binomial"``, rejected otherwise.

    Returns
    -------
    FrequencyModel
        The resolved model.

    Raises
    ------
    ValueError
        If the name is unknown, or ``dispersion`` is missing or supplied where it
        has no meaning.
    """
    if isinstance(frequency, FrequencyModel):
        if dispersion is not None:
            raise ValueError(
                "dispersion= is ignored when frequency is already a FrequencyModel "
                "instance; set it on the model instead."
            )
        return frequency

    name = frequency.strip().lower()
    if name == "poisson":
        if dispersion is not None:
            raise ValueError("dispersion= has no meaning for the Poisson frequency model.")
        return Poisson()
    if name in {"negative_binomial", "negativebinomial", "nb"}:
        if dispersion is None:
            raise ValueError(
                "frequency='negative_binomial' requires dispersion=k, the shape "
                "parameter giving Var(N) = Lambda + Lambda ** 2 / k."
            )
        return NegativeBinomial(dispersion=dispersion)
    raise ValueError(
        f"Unknown frequency model {frequency!r}; expected 'poisson' or 'negative_binomial'."
    )
