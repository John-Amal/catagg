"""Secondary uncertainty: the beta-with-point-mass severity model.

An ELT row states a mean loss ``mu``, a standard deviation ``sigma`` and an
exposed value ``E``. Loss given occurrence is then any distribution on
``[0, E]`` with those two moments. This module picks one by the method of
moments, in the scaled variable ``X / E``.

Write ``m = mu / E`` and ``s = sigma / E``. The method-of-moments beta is::

    nu = m * (1 - m) / s**2 - 1
    alpha = m * nu
    beta = (1 - m) * nu

``nu`` is the concentration ``alpha + beta``, and it is non-negative exactly
when ``s ** 2 <= m * (1 - m)`` — the feasibility bound `catagg.validate_elt`
enforces on ingest. Because validation happens first, the fit never has to clip
a moment it was asked to reproduce.

Where the point masses come from
--------------------------------

The masses at ``0`` and at ``E`` are not free parameters bolted onto the beta.
They are the boundary member of the same family. As ``nu -> 0`` the law
``Beta(m * nu, (1 - m) * nu)`` converges to the Bernoulli law placing ``1 - m``
at ``0`` and ``m`` at ``1``, which is the unique distribution on ``[0, 1]``
attaining the feasibility bound. So the three regimes in `SeverityRegime` are
one continuous family read at three points: a degenerate spike when
``sigma = 0``, an ordinary beta in the interior, and the two-point law on the
endpoints when the variance sits at the bound.

This is a deliberate choice, and it is worth stating plainly. A four-parameter
family — a beta plus a free mass at each endpoint — is the more familiar
picture of secondary uncertainty, and it is what a location-level damage
distribution really looks like. But an ELT row carries two moments, and two
moments cannot determine four parameters. Any interior mass would have to come
from a convention invented here rather than from the data, and a convention
invented here would propagate silently into every tail metric downstream. The
two-parameter fit is exactly determined, reproduces both moments to machine
precision in every regime, and still puts real mass on the endpoints wherever
the moments actually demand it. Use `BetaSeverity.endpoint_masses` to see
where that is. Free endpoint masses belong with a richer input schema, not
with this one.

Numerical note
--------------

For small ``nu`` the beta places all but a vanishing share of its mass within
``exp(-1 / alpha)`` of an endpoint, which underflows float64 long before the
regime boundary is reached. Draws in that range come back as exactly ``0`` or
exactly ``E``. That is float64 representing the fitted distribution faithfully,
not the sampler failing, and the moments are unaffected.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import pandas as pd
from numpy.random import Generator
from numpy.typing import NDArray

from catagg.exceptions import CataggError

# The same relative slack `validate_elt` allows on the feasibility bound. A
# variance that close to the bound is "at the bound" here too: the two must not
# drift apart, or a row could pass validation and then find no fit.
from catagg.io.elt import _VARIANCE_RTOL as _BOUND_RTOL
from catagg.io.elt import validate_elt

__all__ = ["BetaSeverity", "SeverityRegime"]


class SeverityRegime(IntEnum):
    """Which member of the beta-with-point-mass family an event was fitted to.

    Attributes
    ----------
    DEGENERATE
        ``std_loss`` is zero. All mass sits at ``mean_loss``.
    BETA
        The interior case. A beta on ``[0, exposure_value]``, with no atom at
        either endpoint.
    TWO_POINT
        ``std_loss`` is at the feasibility bound. The fit is the two-point law
        carrying mass ``1 - m`` at zero and ``m`` at ``exposure_value``.
    """

    DEGENERATE = 0
    BETA = 1
    TWO_POINT = 2


@dataclass(frozen=True, eq=False)
class BetaSeverity:
    """A method-of-moments severity fit, one distribution per ELT row.

    Build one with `BetaSeverity.from_elt`. Every array is indexed by row
    position in the validated ELT, which is what `catagg.simulate` carries
    through the year loop.

    Attributes
    ----------
    mean_loss
        Target mean loss per event, in loss units.
    exposure_value
        Upper end of the support per event, in loss units.
    alpha
        Beta shape parameter ``alpha``. Zero outside `SeverityRegime.BETA`.
    beta
        Beta shape parameter ``beta``. Zero outside `SeverityRegime.BETA`.
    regime
        `SeverityRegime` code per event, as ``int8``.
    """

    mean_loss: NDArray[np.float64]
    exposure_value: NDArray[np.float64]
    alpha: NDArray[np.float64]
    beta: NDArray[np.float64]
    regime: NDArray[np.int8]

    def __len__(self) -> int:
        """Return the number of fitted events."""
        return int(self.mean_loss.shape[0])

    @classmethod
    def from_elt(cls, elt: pd.DataFrame) -> BetaSeverity:
        """Fit the severity model to every row of an event loss table.

        The ELT is validated first, so the feasibility bound holds and ``nu``
        cannot be meaningfully negative. `catagg.validate_elt` allows a
        relative slack of ``1e-9`` on that bound, so a row written by another
        tool in lower precision can arrive a hair above it; such a row is
        assigned to `SeverityRegime.TWO_POINT`, which reproduces its mean
        exactly and its variance to within that same relative slack.

        Parameters
        ----------
        elt
            Event loss table.

        Returns
        -------
        BetaSeverity
            The fit, with one entry per ELT row in ELT row order.

        Examples
        --------
        >>> import numpy as np
        >>> import catagg as ca
        >>> from catagg.severity import BetaSeverity
        >>> elt = ca.swiss_multiperil_elt(n_events=100, seed=0)
        >>> fit = BetaSeverity.from_elt(elt)
        >>> bool(np.allclose(fit.mean, elt["mean_loss"]))
        True
        """
        elt = validate_elt(elt)
        mean_loss = elt["mean_loss"].to_numpy(dtype=np.float64)
        std_loss = elt["std_loss"].to_numpy(dtype=np.float64)
        exposure_value = elt["exposure_value"].to_numpy(dtype=np.float64)

        n_events = mean_loss.shape[0]
        alpha = np.zeros(n_events, dtype=np.float64)
        beta = np.zeros(n_events, dtype=np.float64)
        regime = np.full(n_events, SeverityRegime.DEGENERATE, dtype=np.int8)

        # nu is only defined where the variance is positive; everything else is
        # a spike at the mean and needs no shape parameters.
        stochastic = std_loss > 0.0
        positions = np.flatnonzero(stochastic)
        scaled_mean = mean_loss[stochastic] / exposure_value[stochastic]
        scaled_std = std_loss[stochastic] / exposure_value[stochastic]
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            nu = scaled_mean * (1.0 - scaled_mean) / scaled_std**2 - 1.0

        candidate_alpha = scaled_mean * nu
        candidate_beta = (1.0 - scaled_mean) * nu

        # Three outcomes. Two guards decide between them, and neither is
        # cosmetic.
        #
        # The concentration is compared against the validator's slack rather
        # than against zero. A row sitting exactly on the feasibility bound
        # computes nu as a few multiples of machine epsilon, of either sign, so
        # a test against zero would label identical rows differently on the last
        # bit - and a fit labelled BETA reports no endpoint mass for a
        # distribution that in practice produces nothing else. Snapping costs at
        # most `_BOUND_RTOL` of relative variance, which is precisely the
        # precision `validate_elt` has already declared it does not resolve.
        #
        # Non-finite shapes mean the scaled variance underflowed: std_loss is
        # positive but so small relative to exposure_value that float64 cannot
        # tell the distribution from a spike. That is a degenerate fit, not a
        # two-point one - routing it to the endpoints would replace a variance
        # of nearly zero with the largest variance the support allows.
        finite = np.isfinite(candidate_alpha) & np.isfinite(candidate_beta)
        interior = finite & (nu > _BOUND_RTOL) & (candidate_alpha > 0.0) & (candidate_beta > 0.0)
        endpoints = finite & ~interior

        alpha[positions[interior]] = candidate_alpha[interior]
        beta[positions[interior]] = candidate_beta[interior]
        regime[positions[interior]] = SeverityRegime.BETA
        regime[positions[endpoints]] = SeverityRegime.TWO_POINT

        return cls(
            mean_loss=mean_loss,
            exposure_value=exposure_value,
            alpha=alpha,
            beta=beta,
            regime=regime,
        )

    def raw_moment(self, order: int) -> NDArray[np.float64]:
        """Return the analytic raw moment ``E[X ** order]`` of each fit.

        Closed form, with no simulation involved. Orders one and two are what
        the fit was built to reproduce; higher orders are what calibrates the
        tolerance of a sampling test, in the same way
        `catagg.aal_standard_error` calibrates the AAL convergence test.

        For the beta regime the ``k``-th raw moment of ``X / E`` is the product
        of ``(alpha + r) / (alpha + beta + r)`` over ``r`` in ``0 .. k - 1``.
        For the two-point regime it is ``m``, for every order, because ``X`` is
        either ``0`` or ``E`` — which is also the limit of the beta product as
        the concentration goes to zero.

        Parameters
        ----------
        order
            Moment order, a strictly positive integer.

        Returns
        -------
        numpy.ndarray
            ``E[X ** order]`` per event, in loss units raised to ``order``.

        Raises
        ------
        ValueError
            If ``order`` is not a strictly positive integer.
        """
        if not isinstance(order, int) or order <= 0:
            raise ValueError(f"order must be a strictly positive integer, got {order!r}.")

        moment = self.mean_loss**order

        two_point = self.regime == SeverityRegime.TWO_POINT
        scaled_mean = np.where(two_point, self.mean_loss / self.exposure_value, 0.0)
        moment = np.where(two_point, scaled_mean * self.exposure_value**order, moment)

        is_beta = self.regime == SeverityRegime.BETA
        alpha = np.where(is_beta, self.alpha, 1.0)
        beta = np.where(is_beta, self.beta, 1.0)
        scaled = np.ones_like(alpha)
        for step in range(order):
            scaled *= (alpha + step) / (alpha + beta + step)
        return np.asarray(
            np.where(is_beta, scaled * self.exposure_value**order, moment), dtype=np.float64
        )

    @property
    def mean(self) -> NDArray[np.float64]:
        """Return the analytic mean of each fit, which must equal ``mean_loss``."""
        return self.raw_moment(1)

    @property
    def std(self) -> NDArray[np.float64]:
        """Return the analytic standard deviation, which must equal ``std_loss``."""
        variance = self.raw_moment(2) - self.raw_moment(1) ** 2
        return np.sqrt(np.maximum(variance, 0.0))

    def endpoint_masses(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the probability mass at zero and at ``exposure_value``.

        Both are zero in `SeverityRegime.BETA`, where the fitted law is
        continuous. They are the whole distribution in `SeverityRegime.TWO_POINT`,
        and a single unit mass in `SeverityRegime.DEGENERATE` when the mean sits
        on an endpoint. See the module docstring for why they are not free
        parameters.

        Returns
        -------
        tuple of numpy.ndarray
            ``(mass at zero, mass at exposure_value)``, per event.
        """
        two_point = self.regime == SeverityRegime.TWO_POINT
        degenerate = self.regime == SeverityRegime.DEGENERATE

        scaled_mean = np.where(two_point, self.mean_loss / self.exposure_value, 0.0)
        at_zero = np.where(two_point, 1.0 - scaled_mean, 0.0)
        at_exposure = np.where(two_point, scaled_mean, 0.0)

        at_zero = np.where(degenerate & (self.mean_loss == 0.0), 1.0, at_zero)
        at_exposure = np.where(
            degenerate & (self.mean_loss == self.exposure_value), 1.0, at_exposure
        )
        return np.asarray(at_zero, dtype=np.float64), np.asarray(at_exposure, dtype=np.float64)

    def sample(self, positions: NDArray[np.int64], rng: Generator) -> NDArray[np.float64]:
        """Draw one loss for each occurrence.

        Occurrences are grouped by regime and each regime is drawn in one
        vectorised call, in the fixed order degenerate, two-point, beta. The
        degenerate group consumes no randomness. Results are reproducible for a
        given fit, position array and seed; the number of uniforms the beta
        sampler consumes per draw is not fixed, so the stream cannot be aligned
        against a different fit.

        Parameters
        ----------
        positions
            Row position into the fit, one per occurrence. Repeats are expected:
            an event occurring in many years appears once per occurrence.
        rng
            Source of randomness.

        Returns
        -------
        numpy.ndarray
            Loss per occurrence, in loss units, within ``[0, exposure_value]``.

        Raises
        ------
        CataggError
            If any position falls outside the fitted events.
        """
        positions = np.asarray(positions, dtype=np.int64)
        if positions.size and (positions.min() < 0 or positions.max() >= len(self)):
            raise CataggError(
                f"severity positions must lie in [0, {len(self)}), got "
                f"[{int(positions.min())}, {int(positions.max())}]."
            )

        losses = np.empty(positions.shape[0], dtype=np.float64)
        regime = self.regime[positions]

        degenerate = np.flatnonzero(regime == SeverityRegime.DEGENERATE)
        losses[degenerate] = self.mean_loss[positions[degenerate]]

        two_point = np.flatnonzero(regime == SeverityRegime.TWO_POINT)
        if two_point.size:
            events = positions[two_point]
            exposure = self.exposure_value[events]
            uniforms = rng.random(two_point.shape[0])
            losses[two_point] = np.where(
                uniforms < self.mean_loss[events] / exposure, exposure, 0.0
            )

        is_beta = np.flatnonzero(regime == SeverityRegime.BETA)
        if is_beta.size:
            events = positions[is_beta]
            losses[is_beta] = self.exposure_value[events] * rng.beta(
                self.alpha[events], self.beta[events]
            )
        return losses
