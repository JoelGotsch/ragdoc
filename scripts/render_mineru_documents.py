"""
Script to render all documents parsed from tests/parsing/data/mineru/*_middle.json files.

Renders each document with all three renderers (raw, prompt/content, embedding)
in Markdown format and writes the output to scripts/output/render_mineru/.

Usage:
    uv run python scripts/render_mineru_documents.py
"""

import json
from pathlib import Path

from ragdoc.parsing.mineru.base import MinerUMiddleDocument
from ragdoc.parsing.mineru.parser import CoreExtractor, MinerUExtractor as MinerUParser
from ragdoc.processing.footnote import SyncFootnoteProcessor
from ragdoc.processing.heading import HeadingLevelProcessor
from ragdoc.rendering import OutputFormat, Renderer, render_for_embedding, render_for_prompt, render_raw

DATA_DIR = Path(__file__).parent.parent / "tests" / "parsing" / "data" / "mineru"
OUTPUT_DIR = Path(__file__).parent / "output" / "render_mineru"

RENDERERS = [
    ("raw", render_raw),
    ("content", render_for_prompt),
    ("embedding", render_for_embedding),
]


def parse_middle_json(path: Path):
    with open(path, encoding="utf-8") as f:
        json_data = json.load(f)
    mineru_doc = MinerUMiddleDocument.model_validate(json_data)
    parser = MinerUParser(use_default_stages=False)
    parser.use(CoreExtractor())
    document = parser.parse_sync(mineru_doc)
    document = HeadingLevelProcessor().process(document)
    document = SyncFootnoteProcessor().process(document)
    return document


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    middle_json_files = sorted(DATA_DIR.glob("*_middle.json"))
    if not middle_json_files:
        raise SystemExit(f"No *_middle.json files found in {DATA_DIR}")

    for json_file in middle_json_files:
        doc_name = json_file.name.replace("_middle.json", "")
        print(f"\nParsing {doc_name} ...")

        document = parse_middle_json(json_file)
        json_out_path = OUTPUT_DIR / f"{doc_name}.json"
        with open(json_out_path, "w", encoding="utf-8") as f:
            json.dump(document.model_dump(), f, ensure_ascii=False, indent=2)
        

        for renderer_name, element_renderer in RENDERERS:
            renderer = Renderer(format=OutputFormat.GFM, element_renderer=element_renderer)
            content = renderer.render(document)

            out_path = OUTPUT_DIR / f"{doc_name}_{renderer_name}.md"
            out_path.write_text(content, encoding="utf-8")
            print(f"  [{renderer_name}] -> {out_path}")

            html_renderer = Renderer(format=OutputFormat.HTML, element_renderer=element_renderer)
            html_content = html_renderer.render(document)
            html_out_path = OUTPUT_DIR / f"{doc_name}_{renderer_name}.html"
            html_out_path.write_text(html_content, encoding="utf-8")
            print(f"  [{renderer_name}] -> {html_out_path}")


    print(f"\nDone. Output written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
