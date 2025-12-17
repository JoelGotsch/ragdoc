"""Tests for FuzzyDate — deterministic derivation of [start, end) + sort_key from EDTF.

Covers the precision matrix (datetime / day / month / year / decade / approximate / range / none),
open intervals (unbounded edges → None), half-open exclusivity, and fail-loud on malformed input.
"""

from __future__ import annotations

import datetime as dt

import pytest
from pydantic import ValidationError

from ragdoc.extraction.dates import FuzzyDate

D = dt.date


def test_exact_datetime():
    fd = FuzzyDate(original_text="12 March 1994, 2pm", edtf="1994-03-12T14:00:00", precision="DATETIME")
    assert fd.precision == "DATETIME"
    assert fd.start == D(1994, 3, 12)
    assert fd.end == D(1994, 3, 13)  # exclusive: day after
    assert fd.sort_key == D(1994, 3, 12)


def test_exact_day():
    fd = FuzzyDate(original_text="1994-03-12", edtf="1994-03-12", precision="DAY")
    assert fd.start == D(1994, 3, 12)
    assert fd.end == D(1994, 3, 13)
    assert fd.sort_key == D(1994, 3, 12)


def test_month():
    fd = FuzzyDate(original_text="March 1994", edtf="1994-03", precision="MONTH")
    assert fd.start == D(1994, 3, 1)
    assert fd.end == D(1994, 4, 1)  # exclusive end = first of next month
    assert fd.sort_key == D(1994, 3, 1)


def test_year():
    fd = FuzzyDate(original_text="1994", edtf="1994", precision="YEAR")
    assert fd.start == D(1994, 1, 1)
    assert fd.end == D(1995, 1, 1)  # exclusive end = first of next year
    assert fd.sort_key == D(1994, 1, 1)


def test_decade():
    fd = FuzzyDate(original_text="the 90s", edtf="199X", precision="DECADE")
    assert fd.precision == "DECADE"
    assert fd.start == D(1990, 1, 1)
    assert fd.end == D(2000, 1, 1)
    assert fd.sort_key == D(1990, 1, 1)


def test_approximate_pads_fuzzy_bounds_but_strict_sort_key():
    fd = FuzzyDate(original_text="circa 1984", edtf="1984~", precision="YEAR")
    # fuzzy bounds widen beyond the strict year; sort_key stays on the strict lower bound.
    assert fd.start == D(1983, 1, 1)
    assert fd.end == D(1986, 1, 1)  # uf = 1985-12-31, exclusive +1 day
    assert fd.sort_key == D(1984, 1, 1)


def test_explicit_range_from_early_decade():
    fd = FuzzyDate(original_text="early 1990s", edtf="1990/1993", precision="YEAR")
    assert fd.start == D(1990, 1, 1)
    assert fd.end == D(1994, 1, 1)
    assert fd.sort_key == D(1990, 1, 1)


def test_no_date_is_unknown_and_unbounded():
    fd = FuzzyDate(original_text="no date here", edtf=None, precision="YEAR")
    assert fd.precision == "UNKNOWN"  # forced even though the LLM guessed YEAR
    assert fd.start is None and fd.end is None and fd.sort_key is None


def test_open_start_interval():
    fd = FuzzyDate(original_text="before 1928", edtf="../1928", precision="YEAR")
    assert fd.start is None  # unbounded earliest
    assert fd.end == D(1929, 1, 1)
    assert fd.sort_key is None


def test_open_end_interval():
    fd = FuzzyDate(original_text="since 1985", edtf="1985/..", precision="YEAR")
    assert fd.start == D(1985, 1, 1)
    assert fd.end is None  # unbounded latest
    assert fd.sort_key == D(1985, 1, 1)


def test_malformed_edtf_raises():
    with pytest.raises(ValidationError):
        FuzzyDate(original_text="whatever", edtf="not-a-date", precision="YEAR")


def test_ascending_hyphen_year_range_normalised_to_interval():
    """US-style '1990-1995' is rewritten to EDTF '1990/1995' and parses as an interval."""
    fd = FuzzyDate(original_text="1990-1995", edtf="1990-1995", precision="YEAR")
    assert fd.edtf == "1990/1995"
    assert fd.start == D(1990, 1, 1)
    assert fd.end == D(1996, 1, 1)  # exclusive end = first year after the interval
    assert fd.sort_key == D(1990, 1, 1)


def test_malformed_hyphen_pair_is_not_rewritten_and_raises():
    """'2024-0312' (a mangled day, not a year range) must NOT become '2024/0312' (year 312,
    reversed interval) — it stays untouched so parse_edtf fails loud (→ LLM retry)."""
    with pytest.raises(ValidationError):
        FuzzyDate(original_text="12 March 2024", edtf="2024-0312", precision="DAY")


def test_non_ascending_hyphen_pair_is_not_rewritten_and_raises():
    """A descending pair like '1995-1990' is not a plausible range; fail loud, don't reorder."""
    with pytest.raises(ValidationError):
        FuzzyDate(original_text="1995-1990", edtf="1995-1990", precision="YEAR")


def test_overlaps_half_open():
    fd = FuzzyDate(original_text="1994", edtf="1994", precision="YEAR")  # [1994-01-01, 1995-01-01)
    # query [1990-01-01, 1996-01-01) overlaps
    assert fd.overlaps(D(1990, 1, 1), D(1996, 1, 1))
    # adjacent query that ends exactly at start does NOT overlap (half-open)
    assert not fd.overlaps(D(1980, 1, 1), D(1994, 1, 1))
    # query that starts exactly at end does NOT overlap
    assert not fd.overlaps(D(1995, 1, 1), D(2000, 1, 1))


def test_no_date_overlaps_anything():
    fd = FuzzyDate(original_text="??", edtf=None, precision="UNKNOWN")
    assert fd.overlaps(D(1990, 1, 1), D(1991, 1, 1))


def test_roundtrip_recomputes_derived_bounds():
    fd = FuzzyDate(original_text="1994", edtf="1994", precision="YEAR")
    reloaded = FuzzyDate.model_validate_json(fd.model_dump_json())
    assert reloaded.start == D(1994, 1, 1) and reloaded.end == D(1995, 1, 1)
