""":class:`FuzzyDate` — a uniform representation of dates of arbitrary precision.

Documents mention dates at wildly varying precision: a full timestamp, a day, a month, a year, a
decade ("the 90s"), an approximate year ("circa 1984"), or an explicit range ("early 1990s"). LLMs
are reliable at mapping such language to a canonical string but measurably **unreliable at date
arithmetic**, so :class:`FuzzyDate` splits the work:

* the **LLM populates** three fields — ``original_text`` (verbatim), ``edtf`` (a canonical
  `Extended Date/Time Format <https://www.loc.gov/standards/datetime/>`_ / ISO 8601-2 string), and
  ``precision`` (a coarse granularity label);
* the library **derives** a half-open ``[start, end)`` interval and a ``sort_key`` deterministically
  from ``edtf`` via the maintained ``edtf`` package, exposed as cached properties.

The derived bounds are ``computed_field`` properties, deliberately **not** input fields. Pydantic
excludes computed fields from the *validation* schema (so the LLM is never asked to produce them —
no schema pollution) but includes them in ``model_dump`` (so they are **persisted** in stores and
become server-side-filterable, e.g. a Qdrant datetime range index). They are recomputed from
``edtf`` on every load, so a stored value is never authoritative. ``start``/``end`` use ``edtf``'s
**fuzzy** bounds (padded for ``~``/``?`` so an approximate date still matches a nearby range query);
``sort_key`` uses the **strict** lower bound (a stable representative instant, since intervals have
no total order).

Range search ("events between 1990 and 1995", i.e. query ``[1990-01-01, 1996-01-01)``) is a
half-open overlap predicate: ``d.start < query_end and d.end > query_start`` (treating ``None`` as
unbounded).
"""

from __future__ import annotations

import datetime as dt
import re
import time
from typing import Literal

from pydantic import BaseModel, Field, PrivateAttr, computed_field, model_validator

_HYPHEN_YEAR_RANGE = re.compile(r"^(\d{4})-(\d{4})$")
_ISO_8601_DURATION = re.compile(r"^P(?:\d+[YMWD])+(?:T(?:\d+[HMS])+)?$|^PT(?:\d+[HMS])+$")


def _normalise_hyphen_year_range(s: str) -> str:
    """Convert ``"YYYY-YYYY"`` (US-style hyphenated year range) to EDTF ``"YYYY/YYYY"``.

    LLMs frequently emit hyphenated ranges from natural-language source text; the EDTF parser
    insists on ``/`` as the range separator. Other hyphenated forms (``2024-03``, ``2024-03-12``)
    are valid EDTF and pass through unchanged because the regex requires exactly four-then-four
    digits separated by a single hyphen.

    Only a **plausible ascending year range** (second year strictly greater than the first) is
    rewritten. Anything else — e.g. ``"2024-0312"``, a mangled day that would otherwise become
    the reversed interval ``2024/0312`` (year 312!) — is returned untouched so ``parse_edtf``
    fails loud and the deterministic validation failure triggers the LLM retry.
    """
    m = _HYPHEN_YEAR_RANGE.match(s.strip())
    if m is None:
        return s
    first, second = m.group(1), m.group(2)
    if int(second) <= int(first):
        return s
    return f"{first}/{second}"


def _is_iso_8601_duration(s: str) -> bool:
    """Return True for ISO 8601 duration strings like ``P2Y9M`` or ``PT1H30M``.

    EDTF (ISO 8601-2) does not include durations — only points and intervals. LLMs sometimes
    emit a duration when the source text describes a span of time (e.g. "two years and nine
    months"). We treat these as "no date" rather than raising, since the LLM has correctly
    identified that there is no calendar reference to anchor.
    """
    return bool(_ISO_8601_DURATION.match(s.strip()))


Precision = Literal["DATETIME", "DAY", "MONTH", "YEAR", "DECADE", "CENTURY", "UNKNOWN"]
"""Coarse granularity of a :class:`FuzzyDate`, set by the LLM (``UNKNOWN`` when no date)."""

_EDTF_IMPORT_ERROR = (
    "FuzzyDate requires the `edtf` package. Install the extraction extra: "
    "`uv sync --extra extraction` (or `pip install ragdoc[extraction]`)."
)


def _to_date(value: object) -> dt.date | None:
    """Convert an ``edtf`` boundary to a :class:`datetime.date`, or ``None`` if unbounded.

    ``edtf`` returns a :class:`time.struct_time` for a bounded edge and a ``float`` (``+/-inf``)
    for an open interval edge (e.g. ``../1928``). The year is clamped into Python's representable
    range so century-scale masks never overflow :class:`datetime.date`.
    """
    if not isinstance(value, time.struct_time):
        return None  # open / unbounded edge (edtf yields +/-inf as a float)
    year = min(max(value.tm_year, dt.MINYEAR), dt.MAXYEAR)
    return dt.date(year, value.tm_mon or 1, value.tm_mday or 1)


def _exclusive_end(inclusive_last: dt.date | None) -> dt.date | None:
    """Return the half-open exclusive end (day after *inclusive_last*), guarding ``date.max``."""
    if inclusive_last is None:
        return None
    if inclusive_last >= dt.date.max:
        return dt.date.max
    return inclusive_last + dt.timedelta(days=1)


class FuzzyDate(BaseModel):
    """A date of arbitrary precision, embeddable in any extraction payload model.

    The LLM fills :attr:`original_text`, :attr:`edtf`, and :attr:`precision`. The interval
    (:attr:`start`/:attr:`end`) and :attr:`sort_key` are derived from :attr:`edtf` and exposed as
    properties — they are not LLM-populated and not stored as fields.
    """

    original_text: str = Field(
        description=(
            "The date EXACTLY as written in the source, verbatim — e.g. 'early 1990s', "
            "'March 1994', 'circa 1984', 'the 90s', '12 March 1994 at 2pm'. Do not normalise it."
        )
    )
    edtf: str | None = Field(
        default=None,
        description=(
            "The date as an Extended Date/Time Format (EDTF / ISO 8601-2) string, or null if no "
            "date is present. Rules:\n"
            "- Exact timestamp: '1994-03-12T14:00:00' (include seconds); keep finer wording in "
            "original_text. Exact day: '1994-03-12'.\n"
            "- Month only: '1994-03'. Year only: '1994'.\n"
            "- A whole decade: mask the last digit — the 1990s -> '199X', the 1980s -> '198X'.\n"
            "- A whole century: the 1900s -> '19XX'.\n"
            "- 'circa'/'about'/'approximately' -> append '~', e.g. '1984~'. 'maybe'/'uncertain' -> "
            "append '?', e.g. '1984?'. Both -> '%'.\n"
            "- Explicit range -> 'start/end', e.g. '1980/1989'. Open end -> '1980/..', open start "
            "-> '../1989'.\n"
            "- For 'early/mid/late <decade>', emit an explicit year RANGE (EDTF cannot narrow "
            "within a decade): 'early 1990s' -> '1990/1993', 'mid 1990s' -> '1994/1996', "
            "'late 1980s' -> '1987/1989'."
        ),
    )
    precision: Precision = Field(
        default="UNKNOWN",
        description=(
            "Granularity of the date: DATETIME (time-of-day known), DAY, MONTH, YEAR, DECADE, "
            "CENTURY, or UNKNOWN (no date present / unparseable). For 'early 1990s' use YEAR (you "
            "emitted an explicit year range); for 'the 90s' use DECADE."
        ),
    )

    _start: dt.date | None = PrivateAttr(default=None)
    _end: dt.date | None = PrivateAttr(default=None)
    _sort_key: dt.date | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _derive_interval(self) -> FuzzyDate:
        """Parse ``edtf`` once and cache the derived bounds; fail loud on a malformed string.

        LLMs often emit US-style hyphenated year ranges (``1990-1991``, ``2004-2005``) instead
        of the EDTF-spec slash form (``1990/1991``). Normalise that one common pattern before
        handing off to the strict parser — saves the otherwise-deterministic Pydantic validation
        failure + LLM retry loop. Other malformed inputs still raise.
        """
        if not self.edtf:
            self.precision = "UNKNOWN"
            return self
        # LLMs sometimes emit an ISO 8601 duration (``P2Y9M``) when source text describes a
        # time span rather than a date. EDTF doesn't include durations — coerce to null edtf
        # + UNKNOWN precision instead of failing validation (which wedges a whole batch).
        if _is_iso_8601_duration(self.edtf):
            self.edtf = None
            self.precision = "UNKNOWN"
            return self
        try:
            from edtf import parse_edtf
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(_EDTF_IMPORT_ERROR) from exc
        edtf_str = _normalise_hyphen_year_range(self.edtf)
        if edtf_str != self.edtf:
            self.edtf = edtf_str  # persist the normalised form
        try:
            obj = parse_edtf(edtf_str)
        except Exception as exc:
            raise ValueError(f"Unparseable EDTF string {self.edtf!r}: {exc}") from exc
        # ``edtf`` ships no usable stubs for the parse result; its bound methods (``lower_strict``
        # etc.) are only reachable on the untyped runtime object.
        self._sort_key = _to_date(obj.lower_strict())  # pyright: ignore[reportCallIssue, reportAttributeAccessIssue, reportOptionalMemberAccess]
        self._start = _to_date(obj.lower_fuzzy())  # pyright: ignore[reportCallIssue, reportAttributeAccessIssue, reportOptionalMemberAccess]
        self._end = _exclusive_end(_to_date(obj.upper_fuzzy()))  # pyright: ignore[reportCallIssue, reportAttributeAccessIssue, reportOptionalMemberAccess]
        return self

    @computed_field
    @property
    def start(self) -> dt.date | None:
        """Inclusive earliest possible date (fuzzy lower bound); ``None`` if unbounded/no date."""
        return self._start

    @computed_field
    @property
    def end(self) -> dt.date | None:
        """Exclusive upper bound of the half-open ``[start, end)`` interval; ``None`` if unbounded."""
        return self._end

    @computed_field
    @property
    def sort_key(self) -> dt.date | None:
        """Representative instant for ordering (strict lower bound); ``None`` if unbounded/no date."""
        return self._sort_key

    def overlaps(self, query_start: dt.date | None, query_end: dt.date | None) -> bool:
        """Return whether this date's interval overlaps the half-open ``[query_start, query_end)``.

        ``None`` bounds are treated as unbounded (``-inf`` for starts, ``+inf`` for ends), so a
        date with no parseable interval (``start`` and ``end`` both ``None``) matches any query.
        """
        if self.end is not None and query_start is not None and self.end <= query_start:
            return False
        if self.start is not None and query_end is not None and self.start >= query_end:
            return False
        return True
