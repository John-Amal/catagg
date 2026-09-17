"""Exception types raised by :mod:`catagg`."""

from __future__ import annotations

__all__ = ["CataggError", "ELTValidationError"]


class CataggError(Exception):
    """Base class for every error raised by catagg."""


class ELTValidationError(CataggError):
    """Raised when an event loss table does not satisfy the ELT schema.

    The message lists *every* failed check, not just the first one, so that a
    malformed table can be fixed in one pass.
    """
