"""
Script to generate OpenAI API response fixtures for LLMFootnoteResolver integration tests.

Run once against a real OpenAI API to produce fixtures, then replay them in tests via a
mock client.  Each fixture captures the exact prompt sent to the model and the raw response
string, so the integration tests validate both the prompt format and the parsing logic
without hitting the network on every run.

Usage:
    uv run python scripts/generate_footnote_resolver_fixtures.py

Requires OPENAI_API_KEY in the environment (or a .env file).
Output:  tests/processing/fixtures/footnote_resolver/
"""

import asyncio
import json
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

from openai import AsyncOpenAI

from ragdoc.processing.footnote import FootnoteCandidate, LLMFootnoteResolver

FIXTURE_DIR = (
    Path(__file__).parent.parent / "tests" / "processing" / "fixtures" / "footnote_resolver"
)

DEFAULT_MODEL = "gpt-4o-mini"


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------


@dataclass
class Scenario:
    name: str
    description: str
    candidates: list[FootnoteCandidate]
    footnote_number: int
    footnote_text: str


def _candidate(
    *,
    page: int,
    context_before: str,
    context_after: str,
    footnote_number: int,
    footnote_text: str,
    element_idx: int,
) -> FootnoteCandidate:
    """Build a FootnoteCandidate with auto-generated IDs."""
    element_id = str(uuid.uuid4())
    footnote_id = str(uuid.uuid4())
    ref_str = str(footnote_number)
    return FootnoteCandidate(
        element_id=element_id,
        element_idx=element_idx,
        page=page,
        match_start=len(context_before),
        match_end=len(context_before) + len(ref_str),
        context_before=context_before,
        context_after=context_after,
        full_context=context_before + ref_str + context_after,
        reference_number=footnote_number,
        footnote_text=footnote_text,
        footnote_id=footnote_id,
    )


def make_scenarios() -> list[Scenario]:
    """Return scenarios covering distinct LLMFootnoteResolver behaviours."""
    return [
        Scenario(
            name="clear_semantic_match",
            description=(
                "Three candidates; only the second has context that clearly relates to the "
                "footnote source (Reuters market report). LLM should return '2'."
            ),
            footnote_number=1,
            footnote_text="Reuters, 'Global Markets Overview', January 2024.",
            candidates=[
                _candidate(
                    page=2,
                    context_before="total return of ",
                    context_after=" year exceeded analyst expectations",
                    footnote_number=1,
                    footnote_text="Reuters, 'Global Markets Overview', January 2024.",
                    element_idx=3,
                ),
                _candidate(
                    page=2,
                    context_before=(
                        "As reported by Reuters, global equity markets declined sharply "
                    ),
                    context_after=" due to rising interest rates and inflation concerns.",
                    footnote_number=1,
                    footnote_text="Reuters, 'Global Markets Overview', January 2024.",
                    element_idx=5,
                ),
                _candidate(
                    page=3,
                    context_before="Figure ",
                    context_after=" illustrates the quarterly distribution of returns.",
                    footnote_number=1,
                    footnote_text="Reuters, 'Global Markets Overview', January 2024.",
                    element_idx=8,
                ),
            ],
        ),
        Scenario(
            name="none_structural_numbers_only",
            description=(
                "All three candidates are structural number references (section, page, figure). "
                "None is a genuine footnote anchor. LLM should return 'NONE'."
            ),
            footnote_number=2,
            footnote_text=(
                "European Banking Authority, 'Guidelines on Capital Requirements', "
                "EBA/GL/2020/14, November 2020."
            ),
            candidates=[
                _candidate(
                    page=4,
                    context_before="see Section ",
                    context_after=" of this report for detailed capital calculations",
                    footnote_number=2,
                    footnote_text=(
                        "European Banking Authority, 'Guidelines on Capital Requirements', "
                        "EBA/GL/2020/14, November 2020."
                    ),
                    element_idx=12,
                ),
                _candidate(
                    page=4,
                    context_before="refer to page ",
                    context_after=" of Annex B for the full regulatory text",
                    footnote_number=2,
                    footnote_text=(
                        "European Banking Authority, 'Guidelines on Capital Requirements', "
                        "EBA/GL/2020/14, November 2020."
                    ),
                    element_idx=15,
                ),
                _candidate(
                    page=4,
                    context_before="Table ",
                    context_after=" summarises the capital adequacy ratios by institution",
                    footnote_number=2,
                    footnote_text=(
                        "European Banking Authority, 'Guidelines on Capital Requirements', "
                        "EBA/GL/2020/14, November 2020."
                    ),
                    element_idx=18,
                ),
            ],
        ),
        Scenario(
            name="literature_citation_mid_sentence",
            description=(
                "Four candidates; the correct one cites 'Johnson' immediately before the "
                "footnote marker, matching the footnote author. LLM should return '3'."
            ),
            footnote_number=3,
            footnote_text="Johnson, M. (2019). 'Systemic Risk and Financial Stability', pp. 45–67.",
            candidates=[
                _candidate(
                    page=1,
                    context_before="revenue declined by ",
                    context_after=" percent compared to the prior year",
                    footnote_number=3,
                    footnote_text="Johnson, M. (2019). 'Systemic Risk and Financial Stability', pp. 45–67.",
                    element_idx=2,
                ),
                _candidate(
                    page=2,
                    context_before="Phase ",
                    context_after=" of the restructuring programme was completed on schedule",
                    footnote_number=3,
                    footnote_text="Johnson, M. (2019). 'Systemic Risk and Financial Stability', pp. 45–67.",
                    element_idx=7,
                ),
                _candidate(
                    page=2,
                    context_before=(
                        "The concept of systemic risk, as defined by Johnson "
                    ),
                    context_after=", has become central to macroprudential policy frameworks.",
                    footnote_number=3,
                    footnote_text="Johnson, M. (2019). 'Systemic Risk and Financial Stability', pp. 45–67.",
                    element_idx=9,
                ),
                _candidate(
                    page=3,
                    context_before="Chart ",
                    context_after=" shows the evolution of the leverage ratio",
                    footnote_number=3,
                    footnote_text="Johnson, M. (2019). 'Systemic Risk and Financial Stability', pp. 45–67.",
                    element_idx=14,
                ),
            ],
        ),
        Scenario(
            name="methodology_footnote_inline",
            description=(
                "Two candidates on the same page; one is at the end of a methodological "
                "statement that naturally calls for a citation. LLM should return '1'."
            ),
            footnote_number=4,
            footnote_text=(
                "Calculated using IFRS 9 expected credit loss (ECL) methodology as described "
                "in the IASB Technical Summary, July 2014."
            ),
            candidates=[
                _candidate(
                    page=5,
                    context_before=(
                        "Loan-loss provisions were computed under the expected credit loss model "
                    ),
                    context_after=" in accordance with applicable accounting standards.",
                    footnote_number=4,
                    footnote_text=(
                        "Calculated using IFRS 9 expected credit loss (ECL) methodology as described "
                        "in the IASB Technical Summary, July 2014."
                    ),
                    element_idx=21,
                ),
                _candidate(
                    page=5,
                    context_before="Q",
                    context_after=" 2023 saw a significant increase in non-performing loans.",
                    footnote_number=4,
                    footnote_text=(
                        "Calculated using IFRS 9 expected credit loss (ECL) methodology as described "
                        "in the IASB Technical Summary, July 2014."
                    ),
                    element_idx=24,
                ),
            ],
        ),
        Scenario(
            name="two_plausible_candidates",
            description=(
                "Two candidates with similar, plausible contexts. The first mentions the "
                "report name; the second references the same institution by acronym. "
                "LLM must pick the stronger contextual match."
            ),
            footnote_number=5,
            footnote_text=(
                "International Monetary Fund, 'World Economic Outlook', April 2023, Chapter 2."
            ),
            candidates=[
                _candidate(
                    page=6,
                    context_before=(
                        "Global GDP growth projections were revised downward by the IMF "
                    ),
                    context_after=" following sustained inflationary pressures in developed markets.",
                    footnote_number=5,
                    footnote_text=(
                        "International Monetary Fund, 'World Economic Outlook', April 2023, Chapter 2."
                    ),
                    element_idx=28,
                ),
                _candidate(
                    page=6,
                    context_before=(
                        "The World Economic Outlook report published earlier this year "
                    ),
                    context_after=" provides the baseline scenario used throughout this analysis.",
                    footnote_number=5,
                    footnote_text=(
                        "International Monetary Fund, 'World Economic Outlook', April 2023, Chapter 2."
                    ),
                    element_idx=31,
                ),
            ],
        ),
    ]


# ---------------------------------------------------------------------------
# Prompt builder — mirrors LLMFootnoteResolver.resolve() exactly
# ---------------------------------------------------------------------------


def _build_user_prompt(
    candidates: list[FootnoteCandidate],
    footnote_number: int,
    footnote_text: str,
) -> str:
    """Reproduce the prompt built inside LLMFootnoteResolver.resolve()."""
    parts = [
        f"Footnote {footnote_number}: {footnote_text}",
        "",
        "Candidate locations:",
    ]
    for i, c in enumerate(candidates, 1):
        parts.append(
            f"{i}. Page {c.page}: ...{c.context_before}[{footnote_number}]{c.context_after}..."
        )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Core fixture generation
# ---------------------------------------------------------------------------


async def generate_fixture(
    scenario: Scenario,
    client: AsyncOpenAI,
    model: str,
) -> dict:
    """
    Call the real OpenAI API for *scenario* and return a fixture dict containing:

    - scenario: scenario name
    - description: human-readable description of what the scenario tests
    - model: model identifier
    - system_prompt: the system message sent to the model
    - user_prompt: the exact user message sent to the model
    - response_content: raw string returned by the model (e.g. "2", "NONE")
    - candidates: serialised FootnoteCandidate list (for documentation / test construction)
    - footnote_number: the footnote number used
    - footnote_text: the footnote definition text used
    """
    user_prompt = _build_user_prompt(
        scenario.candidates,
        scenario.footnote_number,
        scenario.footnote_text,
    )

    print(
        f"\n[{scenario.name}] Sending {len(scenario.candidates)} candidates "
        f"to {model} …"
    )

    response = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": LLMFootnoteResolver.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=10,
    )

    response_content = response.choices[0].message.content.strip()
    print(f"[{scenario.name}] Response: {response_content!r}")

    return {
        "scenario": scenario.name,
        "description": scenario.description,
        "model": model,
        "system_prompt": LLMFootnoteResolver.SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "response_content": response_content,
        "footnote_number": scenario.footnote_number,
        "footnote_text": scenario.footnote_text,
        "candidates": [c.model_dump() for c in scenario.candidates],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY environment variable is not set.")

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    client = AsyncOpenAI(api_key=api_key)
    scenarios = make_scenarios()
    all_fixtures: list[dict] = []

    for scenario in scenarios:
        fixture = await generate_fixture(scenario, client, DEFAULT_MODEL)
        all_fixtures.append(fixture)

        path = FIXTURE_DIR / f"{scenario.name}.json"
        path.write_text(json.dumps(fixture, indent=2), encoding="utf-8")
        print(f"[{scenario.name}] Saved → {path}")

    # Combined index for convenience
    index_path = FIXTURE_DIR / "index.json"
    index = [
        {
            "scenario": f["scenario"],
            "description": f["description"],
            "file": f"{f['scenario']}.json",
            "candidate_count": len(f["candidates"]),
            "footnote_number": f["footnote_number"],
            "response_content": f["response_content"],
        }
        for f in all_fixtures
    ]
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"\nAll done. {len(all_fixtures)} fixtures written to {FIXTURE_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
