"""Catastrophe loss aggregation, from event loss tables to net exceedance probability curves."""

from __future__ import annotations

from catagg.examples import swiss_multiperil_elt
from catagg.exceptions import CataggError, ELTValidationError
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
from catagg.metrics import aal_standard_error, analytic_aal, analytic_annual_loss_variance
from catagg.simulate import simulate, to_ylt

__version__ = "0.1.0.dev0"

__all__ = [
    "ELT_COLUMNS",
    "ELT_DTYPES",
    "YELT_COLUMNS",
    "YELT_DTYPES",
    "CataggError",
    "ELTValidationError",
    "FrequencyModel",
    "NegativeBinomial",
    "Poisson",
    "__version__",
    "aal_standard_error",
    "analytic_aal",
    "analytic_annual_loss_variance",
    "read_elt",
    "read_yelt",
    "simulate",
    "swiss_multiperil_elt",
    "to_ylt",
    "validate_elt",
    "write_yelt",
]
