"""Reinsurance structures: quota share, excess of loss with reinstatements, stop-loss."""

from __future__ import annotations

from catagg.financial.programme import Programme, ProgrammeResult
from catagg.financial.structures import AggregateStopLoss, ExcessOfLoss, QuotaShare, Structure

__all__ = [
    "AggregateStopLoss",
    "ExcessOfLoss",
    "Programme",
    "ProgrammeResult",
    "QuotaShare",
    "Structure",
]
