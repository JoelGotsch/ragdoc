"""
Script to generate LLM response fixtures for heading_llm integration tests.

Run once against a real OpenAI API to produce fixtures, then use the saved
fixtures in tests via a mock client that replays the recorded responses.

Usage:
    uv run python scripts/generate_heading_llm_fixtures.py

Requires OPENAI_API_KEY in the environment (or a .env file).
Output:  tests/processing/fixtures/heading_llm/
"""

import asyncio
import json
from pathlib import Path

from openai import AsyncOpenAI

from ragdoc.document import Document
from ragdoc.parsing.mineru.base import MinerUMiddleDocument
from ragdoc.parsing.mineru.parser import CoreExtractor, MinerUExtractor as MinerUParser
from ragdoc.processing.heading import HeadingLevelProcessor
from ragdoc.processing.heading_llm import HeadingResponse, LLMHeadingResolver, LLMHeadingResolverSettings

FIXTURE_DIR = Path(__file__).parent.parent / "tests" / "processing" / "fixtures" / "heading_llm"
TEST_CASES_FILE = Path(__file__).parent.parent / "tests" / "data" / "test_cases.json"
MINERU_DIR = Path(__file__).parent.parent / "tests" / "parsing" / "data" / "mineru"


# ---------------------------------------------------------------------------
# Test scenarios – each is a (name, Document) pair loaded from test_cases.json
# ---------------------------------------------------------------------------

async def make_scenarios() -> list[tuple[str, Document]]:
    """Return (scenario_name, document) pairs loaded from tests/data/test_cases.json."""
    with open(TEST_CASES_FILE, encoding="utf-8") as f:
        cases = json.load(f)

    parser = MinerUParser(use_default_stages=False)
    parser.use(CoreExtractor())

    scenarios: list[tuple[str, Document]] = []
    for name in cases:
        path = MINERU_DIR / f"{name}_middle.json"
        if not path.exists():
            print(f"[{name}] Skipping — no mineru file at {path}")
            continue
        with open(path, encoding="utf-8") as f:
            middle = MinerUMiddleDocument.model_validate(json.load(f))
        doc = await parser.parse(middle)
        doc = HeadingLevelProcessor(trust_parser_levels=False).process(doc)
        scenarios.append((name, doc))

    return scenarios


# ---------------------------------------------------------------------------
# Core fixture generation logic
# ---------------------------------------------------------------------------

async def generate_fixture(
    name: str,
    document: Document,
    resolver: LLMHeadingResolver,
) -> dict:
    """
    Call the real OpenAI API for *document* and return a fixture dict containing:

    - scenario: scenario name
    - heading_infos: serialised HeadingInfo list (for documentation)
    - user_prompt: exact prompt sent to the model
    - system_prompt: the system message used
    - response_content: raw JSON string returned by the model
    - model: model identifier
    """
    client = resolver.client
    settings = resolver.settings
    heading_infos = resolver._collect_heading_infos(document)
    user_prompt = resolver._build_prompt(heading_infos)

    print(f"\n[{name}] Sending {len(heading_infos)} headings to {settings.model_name} …")

    response = await client.beta.chat.completions.parse(
        model=settings.model_name,
        messages=[
            {"role": "system", "content": LLMHeadingResolver.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=settings.temperature,
        response_format=HeadingResponse,
    )

    parsed = response.choices[0].message.parsed
    if parsed is None:
        raise ValueError(f"[{name}] LLM refused to provide judgments")
    response_content = parsed.model_dump_json()
    print(f"[{name}] Response: {response_content}")

    return {
        "scenario": name,
        "model": settings.model_name,
        "system_prompt": LLMHeadingResolver.SYSTEM_PROMPT,
        "user_prompt": user_prompt,
        "response_content": response_content,
        "heading_infos": [
            {
                "index": hi.index,
                "page_number": hi.page_number,
                "font_size": hi.font_size,
                "is_centered": hi.is_centered,
                "text": hi.text,
            }
            for hi in heading_infos
        ],
    }


async def main() -> None:
    # api_key = os.environ.get("OPENAI_API_KEY")
    # if not api_key:
    #     raise SystemExit("OPENAI_API_KEY environment variable is not set.")

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    from httpx import AsyncClient

    settings = LLMHeadingResolverSettings()
    http_client = AsyncClient(verify=False) # str(Path(__file__).parent.parent / 'certs' / 'ca-bundle-all.crt'))
    openai_client = AsyncOpenAI(
        max_retries=3,
        api_key=settings.api_key.get_secret_value(),
        base_url=settings.base_url,
        http_client=http_client,
    )
    # resolver = LLMHeadingResolver(client, settings)
    resolver = LLMHeadingResolver(client=openai_client)

    scenarios = await make_scenarios()
    all_fixtures: list[dict] = []

    for name, document in scenarios:
        fixture = await generate_fixture(name, document, resolver)
        all_fixtures.append(fixture)

        # Write one file per scenario for easy individual use
        path = FIXTURE_DIR / f"{name}.json"
        path.write_text(json.dumps(fixture, indent=2), encoding="utf-8")
        print(f"[{name}] Saved → {path}")

    # Also write a combined index for convenience
    index_path = FIXTURE_DIR / "index.json"
    index = [
        {
            "scenario": f["scenario"],
            "file": f"{f['scenario']}.json",
            "heading_count": len(f["heading_infos"]),
            "response_content": f["response_content"],
        }
        for f in all_fixtures
    ]
    index_path.write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"\nAll done. {len(all_fixtures)} fixtures written to {FIXTURE_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
