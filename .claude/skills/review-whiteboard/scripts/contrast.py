"""WCAG contrast gate for review pages: text colours and meaningful (dashed) strokes.

Text must reach 4.5 : 1 against both the page paper and white panels. A dashed stroke carries
meaning (lifeline, fragment, removed frame) and must reach 3 : 1, which for the ink colour means a
``stroke-opacity`` of at least 0.55. Solid hairlines are decoration and are not checked.
"""

from __future__ import annotations

import re

BACKGROUNDS = ("#ffffff", "#f5f5f5")
MIN_TEXT = 4.5
MIN_DASHED_OPACITY = 0.55

STYLE_RE = re.compile(r"<style\b[^>]*>(.*?)</style>", re.DOTALL)
RULE_RE = re.compile(r"([^{}]+)\{([^{}]*)\}")
TOKEN_RE = re.compile(r"(--[\w-]+)\s*:\s*([^;]+);")
DECL_RE = re.compile(r"(?<![\w-])(color|fill)\s*:\s*([^;]+)")
TEXT_TAG_RE = re.compile(r"<(?:text|tspan)\b[^>]*\bfill=\"([^\"]+)\"")
DASHED_TAG_RE = re.compile(r"<(?:line|path|rect|polygon|polyline|circle|ellipse)\b[^>]*>")
HEX_RE = re.compile(r"^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")


def luminance(hex_colour: str) -> float:
    value = hex_colour.lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    channels = [int(value[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    linear = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def contrast(a: str, b: str) -> float:
    high, low = sorted((luminance(a), luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def resolve_colour(value: str, tokens: dict[str, str]) -> str | None:
    """A hex colour for ``value`` (following ``var(--x)``), or None for rgba/inherit/currentColor."""
    value = value.strip().removesuffix("!important").strip()
    for _ in range(5):
        var = re.fullmatch(r"var\((--[\w-]+)(?:\s*,[^)]*)?\)", value)
        if var is None:
            break
        value = tokens.get(var.group(1), "").strip()
    return value if HEX_RE.match(value) else None


def worst_contrast(colour: str) -> float:
    return min(contrast(colour, background) for background in BACKGROUNDS)


def text_findings(page: str) -> list[str]:
    styles = STYLE_RE.findall(page)
    tokens = {name: value for css in styles for name, value in TOKEN_RE.findall(css)}
    findings: list[str] = []
    for css in styles:
        css = re.sub(r"/\*.*?\*/", "", css, flags=re.DOTALL)
        for selector, body in RULE_RE.findall(css):
            for prop, raw in DECL_RE.findall(body):
                if prop == "fill" and "font" not in body:
                    continue  # a shape fill, not text
                colour = resolve_colour(raw, tokens)
                if colour and worst_contrast(colour) < MIN_TEXT:
                    findings.append(
                        f"low-contrast text: `{selector.strip()}` {prop} {raw.strip()} is "
                        f"{worst_contrast(colour):.1f}:1, needs {MIN_TEXT}:1 (use --color-faint or darker)"
                    )
    for raw in dict.fromkeys(TEXT_TAG_RE.findall(page)):
        colour = resolve_colour(raw, tokens)
        if colour and worst_contrast(colour) < MIN_TEXT:
            findings.append(
                f'low-contrast text: <text fill="{raw}"> is {worst_contrast(colour):.1f}:1, needs {MIN_TEXT}:1'
            )
    return findings


def stroke_findings(page: str) -> list[str]:
    findings: list[str] = []
    for tag in DASHED_TAG_RE.findall(page):
        if "stroke-dasharray" not in tag:
            continue
        opacity = re.search(r'stroke-opacity="([\d.]+)"', tag)
        if opacity and float(opacity.group(1)) < MIN_DASHED_OPACITY:
            findings.append(
                f"faint dashed stroke (stroke-opacity {opacity.group(1)} < {MIN_DASHED_OPACITY}): {tag[:90]}…"
            )
    return findings


def contrast_findings(page: str) -> list[str]:
    return text_findings(page) + stroke_findings(page)
