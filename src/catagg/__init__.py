"""Catastrophe loss aggregation, from event loss tables to net exceedance probability curves."""

from __future__ import annotations

from catagg.examples import swiss_multiperil_elt
from catagg.exceptions import CataggError, ELTValidationError
from catagg.financial import (
    AggregateStopLoss,
    ExcessOfLoss,
    Programme,
    ProgrammeResult,
    QuotaShare,
    Structure,
)
from catagg.frequency import FrequencyModel, NegativeBinomial, Poisson
from catagg.io import (
    ELT_COLUMNS,
    ELT_DTYPES,
    YELT_COLUMNS,
    YELT_DTYPES,
    read_elt,
    read_yelt,
    validate_elt,
    write_yelt,
)
from catagg.metrics import (
    STANDARD_RETURN_PERIODS,
    EPCurve,
    aal_standard_error,
    analytic_aal,
    analytic_annual_loss_variance,
    ep_curve,
)
from catagg.severity import BetaSeverity, SeverityRegime
from catagg.simulate import simulate, to_ylt

__version__ = "0.1.0.dev1"

__all__ = [
    "ELT_COLUMNS",
    "ELT_DTYPES",
    "STANDARD_RETURN_PERIODS",
    "YELT_COLUMNS",
    "YELT_DTYPES",
    "AggregateStopLoss",
    "BetaSeverity",
    "CataggError",
    "ELTValidationError",
    "EPCurve",
    "ExcessOfLoss",
    "FrequencyModel",
    "NegativeBinomial",
    "Poisson",
    "Programme",
    "ProgrammeResult",
    "QuotaShare",
    "SeverityRegime",
    "Structure",
    "__version__",
    "aal_standard_error",
    "analytic_aal",
    "analytic_annual_loss_variance",
    "ep_curve",
    "read_elt",
    "read_yelt",
    "simulate",
    "swiss_multiperil_elt",
    "to_ylt",
    "validate_elt",
    "write_yelt",
]
