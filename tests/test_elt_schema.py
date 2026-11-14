"""Tests for the ELT schema and its validator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from catagg import ELT_COLUMNS, ELTValidationError, read_elt, validate_elt


def test_valid_elt_is_normalised(tiny_elt: pd.DataFrame) -> None:
    validated = validate_elt(tiny_elt)
    assert list(validated.columns) == list(ELT_COLUMNS)
    assert validated["event_id"].dtype == np.dtype("int64")
    assert validated["rate"].dtype == np.dtype("float64")
    assert str(validated["peril"].dtype) == "string"


def test_validation_does_not_mutate_input(tiny_elt: pd.DataFrame) -> None:
    before = tiny_elt.copy(deep=True)
    validate_elt(tiny_elt)
    pd.testing.assert_frame_equal(tiny_elt, before)


def test_extra_columns_are_preserved_after_canonical_ones(tiny_elt: pd.DataFrame) -> None:
    tiny_elt["footprint_id"] = [10, 20, 30, 40, 50]
    validated = validate_elt(tiny_elt)
    assert list(validated.columns) == [*ELT_COLUMNS, "footprint_id"]


def test_missing_columns_are_all_named(tiny_elt: pd.DataFrame) -> None:
    with pytest.raises(ELTValidationError) as error:
        validate_elt(tiny_elt.drop(columns=["std_loss", "region"]))
    assert "std_loss" in str(error.value)
    assert "region" in str(error.value)


@pytest.mark.parametrize(
    ("column", "value", "fragment"),
    [
        ("rate", 0.0, "rate"),
        ("rate", -0.1, "rate"),
        ("rate", np.nan, "rate"),
        ("rate", np.inf, "rate"),
        ("mean_loss", -1.0, "mean_loss"),
        ("std_loss", -1.0, "std_loss"),
        ("exposure_value", 0.0, "exposure_value"),
    ],
)
def test_out_of_range_values_are_rejected(
    tiny_elt: pd.DataFrame, column: str, value: float, fragment: str
) -> None:
    tiny_elt.loc[2, column] = value
    with pytest.raises(ELTValidationError, match=fragment):
        validate_elt(tiny_elt)


def test_duplicate_event_ids_are_rejected(tiny_elt: pd.DataFrame) -> None:
    tiny_elt.loc[4, "event_id"] = 1
    with pytest.raises(ELTValidationError, match="unique"):
        validate_elt(tiny_elt)


def test_mean_loss_above_exposure_is_rejected(tiny_elt: pd.DataFrame) -> None:
    tiny_elt.loc[0, "mean_loss"] = tiny_elt["exposure_value"].to_numpy()[0] * 1.01
    with pytest.raises(ELTValidationError, match="exposure_value"):
        validate_elt(tiny_elt)


def test_std_loss_at_the_variance_bound_is_accepted(tiny_elt: pd.DataFrame) -> None:
    """The two-point distribution on {0, exposure} is feasible, so its std must pass."""
    mean = tiny_elt["mean_loss"].to_numpy()[0]
    exposure = tiny_elt["exposure_value"].to_numpy()[0]
    tiny_elt.loc[0, "std_loss"] = np.sqrt(mean * (exposure - mean))
    validate_elt(tiny_elt)


def test_std_loss_above_the_variance_bound_is_rejected(tiny_elt: pd.DataFrame) -> None:
    mean = tiny_elt["mean_loss"].to_numpy()[0]
    exposure = tiny_elt["exposure_value"].to_numpy()[0]
    tiny_elt.loc[0, "std_loss"] = np.sqrt(mean * (exposure - mean)) * 1.001
    with pytest.raises(ELTValidationError, match="std_loss exceeds"):
        validate_elt(tiny_elt)


def test_empty_peril_label_is_rejected(tiny_elt: pd.DataFrame) -> None:
    tiny_elt.loc[3, "peril"] = "   "
    with pytest.raises(ELTValidationError, match="peril"):
        validate_elt(tiny_elt)


def test_all_failures_are_reported_together(tiny_elt: pd.DataFrame) -> None:
    tiny_elt.loc[0, "rate"] = -1.0
    tiny_elt.loc[1, "mean_loss"] = -1.0
    tiny_elt.loc[[2], "event_id"] = tiny_elt.loc[[3], "event_id"].to_numpy()
    with pytest.raises(ELTValidationError) as error:
        validate_elt(tiny_elt)
    message = str(error.value)
    assert "rate" in message
    assert "mean_loss" in message
    assert "unique" in message


def test_roundtrip_through_parquet_and_csv(tiny_elt: pd.DataFrame, tmp_path: Path) -> None:
    validated = validate_elt(tiny_elt)
    for suffix in (".parquet", ".csv"):
        path = tmp_path / f"elt{suffix}"
        if suffix == ".parquet":
            validated.to_parquet(path)
        else:
            validated.to_csv(path, index=False)
        pd.testing.assert_frame_equal(read_elt(path), validated)


def test_unknown_suffix_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "elt.txt"
    path.write_text("nothing")
    with pytest.raises(ValueError, match="Unsupported ELT format"):
        read_elt(path)
