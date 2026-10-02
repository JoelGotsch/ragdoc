from __future__ import annotations

import json
import re
import unicodedata

from pydantic import BaseModel


def escape_markdown(text: str) -> str:
    escape_chars = r"_*[]()~`>#+-=|{}.!&$"
    return re.sub(f"([{re.escape(escape_chars)}])", r"\\\1", text)


def add_unregister(func):
    # build a dictionary mapping names to closure cells
    closure = dict(zip(func.register.__code__.co_freevars, func.register.__closure__, strict=True))
    registry = closure["registry"].cell_contents
    dispatch_cache = closure["dispatch_cache"].cell_contents

    def unregister(cls):
        del registry[cls]
        dispatch_cache.clear()

    func.unregister = unregister
    return func


class MetadataEncoder(json.JSONEncoder):
    """JSON encoder that serializes Pydantic ``BaseModel`` instances via ``model_dump()``.

    Plain ``json.dumps()`` raises ``TypeError`` when it encounters a BaseModel
    inside a dict or list.  This encoder intercepts those objects and converts
    them to plain dicts first, so arbitrarily nested structures like
    ``{"key": SomeModel(...)}`` serialize correctly.
    """

    def default(self, o: object) -> object:
        if isinstance(o, BaseModel):
            return o.model_dump()
        return super().default(o)


def normalize_text(text: str) -> str:
    """Normalize text for comparison (whitespace + Unicode NFKC).

    Useful for aligning visually identical text such as ``H₂O`` vs ``H2O``.
    """
    # Strip single-tilde markdown subscripts (e.g., H~2~O -> H2O) with a tight, short payload to avoid altering other tilde uses
    text = re.sub(r"~([0-9A-Za-z]{1,10})~", r"\1", text)
    normalized = unicodedata.normalize("NFKC", text).casefold()
    normalized = re.sub(r"\s+", " ", normalized).strip()
    # Normalize all quote-like characters to straight single quote:
    # - Straight double quote: " (U+0022)
    # - Curly double quotes: " " (U+201C, U+201D)
    # - Curly single quotes/apostrophes: ' ' (U+2018, U+2019)
    # - Low quotes: „ ‚ (U+201E, U+201A)
    # - Guillemets: « » ‹ › (U+00AB, U+00BB, U+2039, U+203A)
    # - Prime symbols: ′ ″ (U+2032, U+2033)
    # - Grave/acute accents: ` ´ (U+0060, U+00B4)
    normalized = re.sub(
        r'["\u201C\u201D\u201E\'\u2018\u2019\u201A\u00AB\u00BB\u2039\u203A\u2032\u2033`\u00B4]', "'", normalized
    )
    # Strip trailing punctuation (periods) for comparison
    normalized = normalized.strip(".")
    return f"{normalized}"
