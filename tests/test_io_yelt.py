"""YELT persistence and the synthetic example generator."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from catagg import CataggError, read_yelt, simulate, swiss_multiperil_elt, validate_elt, write_yelt


def test_yelt_roundtrips_through_parquet_with_its_year_count(
    tiny_elt: pd.DataFrame, tmp_path: Path
) -> None:
    yelt = simulate(tiny_elt, n_years=5_000, seed=61)
    path = tmp_path / "yelt.parquet"
    write_yelt(yelt, path)
    restored = read_yelt(path)
    pd.testing.assert_frame_equal(restored, yelt)
    assert restored.attrs["n_years"] == 5_000


def test_writing_needs_a_year_count_when_attrs_are_lost(tiny_elt: pd.DataFrame, tmp_path) -> None:  # type: ignore[no-untyped-def]
    yelt = simulate(tiny_elt, n_years=1_000, seed=62)
    yelt.attrs.clear()
    with pytest.raises(CataggError, match="n_years is unknown"):
        write_yelt(yelt, tmp_path / "yelt.parquet")
    write_yelt(yelt, tmp_path / "yelt.parquet", n_years=1_000)
    assert read_yelt(tmp_path / "yelt.parquet").attrs["n_years"] == 1_000


def test_reading_a_foreign_parquet_file_is_refused(tiny_elt: pd.DataFrame, tmp_path) -> None:  # type: ignore[no-untyped-def]
    path = tmp_path / "foreign.parquet"
    simulate(tiny_elt, n_years=100, seed=63).to_parquet(path)
    with pytest.raises(CataggError, match=r"catagg\.n_years"):
        read_yelt(path)


class TestExampleGenerator:
    def test_output_is_a_valid_elt(self) -> None:
        elt = swiss_multiperil_elt(n_events=200, seed=3)
        pd.testing.assert_frame_equal(validate_elt(elt), elt)

    def test_peril_rates_sum_to_the_specification(self) -> None:
        from catagg.examples import SWISS_PERILS

        elt = swiss_multiperil_elt(n_events=500, seed=3)
        totals = elt.groupby("peril", observed=True)["rate"].sum()
        for spec in SWISS_PERILS:
            assert totals[spec.name] == pytest.approx(spec.annual_rate)

    def test_generation_is_reproducible(self) -> None:
        pd.testing.assert_frame_equal(
            swiss_multiperil_elt(n_events=100, seed=9), swiss_multiperil_elt(n_events=100, seed=9)
        )

    def test_non_positive_event_count_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="n_events"):
            swiss_multiperil_elt(n_events=0)
