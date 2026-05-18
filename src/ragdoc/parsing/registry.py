"""Parser registry — maps file suffix patterns to async parsers.

The registry is the backbone of :func:`~ragdoc.parsing.load`.  Parsers are
registered with :func:`register_parser` and looked up via :func:`_resolve_parser`.

Resolution order (highest to lowest precedence):
1. Explicit ``parser=`` argument on :func:`load`.
2. ``parser_preferences`` from the active :class:`~ragdoc.config.RagdocConfig`.
3. Longest-matching suffix pattern, then highest priority number.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from ragdoc.parsing.parser import Parser


class ParserRegistration(NamedTuple):
    """A single entry in the parser registry."""

    pattern: str
    parser: "Parser"
    priority: int
    name: str


_registry: list[ParserRegistration] = []


def register_parser(parser: "Parser") -> None:
    """Register a parser for all its declared patterns.

    Patterns, name, priority, and description are read from the parser instance.
    Each pattern becomes its own registry entry (suffix-based lookup is per-pattern).

    Args:
        parser: A :class:`~ragdoc.parsing.parser.Parser` instance.  Its
            ``patterns``, ``name``, ``priority``, and ``description`` fields
            drive registration.
    """
    for pattern in parser.patterns:
        _registry.append(
            ParserRegistration(
                pattern=pattern,
                parser=parser,
                priority=parser.priority,
                name=parser.name,
            )
        )


def unregister_parser(pattern: str, name: str) -> None:
    """Remove a specific parser registration.

    Args:
        pattern: The suffix pattern of the registration to remove.
        name: The name of the registration to remove.

    Raises:
        ValueError: If no matching registration is found.
    """
    for i, reg in enumerate(_registry):
        if reg.pattern == pattern and reg.name == name:
            _registry.pop(i)
            return
    raise ValueError(f"No parser registration found for pattern={pattern!r}, name={name!r}")


def get_parser(name: str) -> "Parser":
    """Look up a registered parser by name.

    Useful for constructing :class:`~ragdoc.parsing.multi_source.MultiSourceParser`
    from named registrations.

    Args:
        name: The registration name to look up (exact match).

    Returns:
        The registered :class:`~ragdoc.parsing.parser.Parser`.

    Raises:
        ValueError: If no parser is registered with the given name.
    """
    for reg in _registry:
        if reg.name == name:
            return reg.parser
    available = sorted({r.name for r in _registry})
    raise ValueError(f"No parser registered with name={name!r}. Available: {available}")


def get_registered_parsers() -> list[ParserRegistration]:
    """Return all registrations (for introspection/debugging)."""
    return list(_registry)


def describe_registry() -> str:
    """Return a human-readable summary of all registered parsers.

    Groups registrations by parser name and lists the suffix patterns each
    parser handles.  Priority is shown as ``[p=N]`` only when non-zero.

    Example output::

        Registered parsers:
          html         .html .htm             HTML files
          pandoc       .docx .doc             Word documents via pandoc
          azure_di     .pdf       [p=40]      PDF via Azure Document Intelligence
    """
    if not _registry:
        return "No parsers registered."

    # Group by parser name, preserving first-seen insertion order
    groups: dict[str, list[ParserRegistration]] = defaultdict(list)
    for reg in _registry:
        groups[reg.name].append(reg)

    rows: list[tuple[str, str, str, str]] = []
    for name, regs in groups.items():
        patterns = sorted({r.pattern for r in regs}, key=lambda p: (-len(p), p))
        priority = regs[0].priority
        description = regs[0].parser.description
        pattern_str = " ".join(patterns)
        priority_str = f"[p={priority}]" if priority > 0 else ""
        rows.append((name, pattern_str, priority_str, description))

    name_w = max(len(r[0]) for r in rows) + 2
    pat_w = max(len(r[1]) for r in rows) + 2
    pri_w = max(len(r[2]) for r in rows) + 1 if any(r[2] for r in rows) else 0

    lines = ["Registered parsers:"]
    for name, pattern_str, priority_str, description in rows:
        if pri_w:
            line = f"  {name:<{name_w}}{pattern_str:<{pat_w}}{priority_str:<{pri_w}}  {description}"
        else:
            line = f"  {name:<{name_w}}{pattern_str:<{pat_w}}  {description}"
        lines.append(line.rstrip())

    return "\n".join(lines)


def _resolve_parser(path: Path, parser_name: str | None = None) -> "Parser":
    """Find the best parser for a path.

    Resolution:
    1. If *parser_name* is given, find registration by name (exact match).
    2. Else check ``config.parser_preferences`` for a name override.
    3. Else find all registrations where ``path.name`` ends with the pattern,
       sort by ``(-len(pattern), -priority)``, return first.
    4. Raise ``ValueError`` if no match.
    """
    if parser_name is not None:
        return get_parser(parser_name)

    # Check config preferences
    from ragdoc.config import get_config

    config = get_config()
    preferences: dict[str, str] = getattr(config, "parser_preferences", {})
    filename = path.name.lower()
    # Check preferences: longest matching preference pattern first
    if preferences:
        matching_prefs = [
            (pat, pname) for pat, pname in preferences.items() if filename.endswith(pat.lower())
        ]
        if matching_prefs:
            matching_prefs.sort(key=lambda x: -len(x[0]))
            return get_parser(matching_prefs[0][1])

    # Find all matching registrations by suffix pattern
    matches = [reg for reg in _registry if filename.endswith(reg.pattern.lower())]
    if not matches:
        available_patterns = sorted({r.pattern for r in _registry})
        raise ValueError(
            f"No parser registered for {path.name!r}. "
            f"Available patterns: {available_patterns}"
        )

    # Sort by longest pattern first, then highest priority
    matches.sort(key=lambda r: (-len(r.pattern), -r.priority))
    return matches[0].parser


def _clear_registry() -> None:
    """Remove all registrations. For testing only."""
    _registry.clear()
