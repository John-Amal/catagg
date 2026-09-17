"""Shared fixtures."""

from __future__ import annotations

import pandas as pd
import pytest

from catagg.examples import swiss_multiperil_elt


@pytest.fixture
def tiny_elt() -> pd.DataFrame:
    """Build a five-event, two-peril ELT with hand-checkable rates and means.

    Total rate 1.5, analytic AAL 1.5e6 + ... deliberately left to the tests to
    compute, so that a change here cannot silently agree with a wrong expectation.
    """
    return pd.DataFrame(
        {
            "event_id": [1, 2, 3, 4, 5],
            "rate": [0.50, 0.25, 0.10, 0.60, 0.05],
            "mean_loss": [1.0e6, 4.0e6, 2.0e7, 5.0e5, 8.0e7],
            "std_loss": [5.0e5, 2.0e6, 1.0e7, 2.5e5, 4.0e7],
            "exposure_value": [5.0e6, 2.0e7, 1.0e8, 2.5e6, 4.0e8],
            "peril": ["hail", "hail", "flood", "flood", "windstorm"],
            "region": ["Plateau", "Alpine", "Plateau", "Jura", "Plateau"],
        }
    )


@pytest.fixture
def example_elt() -> pd.DataFrame:
    """Build a 300-event synthetic three-peril portfolio."""
    return swiss_multiperil_elt(n_events=300, seed=7)
