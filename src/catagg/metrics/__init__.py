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

__all__ = ["aal_standard_error", "analytic_aal", "analytic_annual_loss_variance"]
