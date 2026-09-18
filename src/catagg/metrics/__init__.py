"""EP curves, TVaR, AAL, and Monte Carlo standard errors.

Only the closed-form reference quantities are implemented so far; the simulated
EP curve machinery lands in a later milestone.
"""

from __future__ import annotations

from catagg.metrics.analytic import (
    aal_standard_error,
    analytic_aal,
    analytic_annual_loss_variance,
)
from catagg.metrics.ep import STANDARD_RETURN_PERIODS, EPCurve, ep_curve

__all__ = [
    "STANDARD_RETURN_PERIODS",
    "EPCurve",
    "aal_standard_error",
    "analytic_aal",
    "analytic_annual_loss_variance",
    "ep_curve",
]
