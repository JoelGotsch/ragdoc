import marimo

__generated_with = "0.23.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Extraction & Knowledge Graphs

    Structured extraction turns documents into **typed records** synced incrementally, in
    three layers:

    1. **Mentions** — raw, provenance-tagged occurrences, produced by an
       [`Extractor`][ragdoc.extraction.Extractor] and synced per source by
       [`MentionStorePipeline`][ragdoc.extraction.MentionStorePipeline].
    2. **Entities** — canonical, deduplicated records, produced by
       [`EntityResolutionPipeline`][ragdoc.extraction.EntityResolutionPipeline].
    3. **Knowledge graph** — typed nodes + edges against a
       [`GraphSchema`][ragdoc.extraction.GraphSchema] (see the
       [Extraction Guide](../guide/extraction.md)).

    This notebook runs the mention → entity workflow end-to-end **without an API key**:
    the `Extractor` protocol is structural, so a deterministic stub stands in for the
    LLM-backed `StructuredExtractor`, and the resolution collaborators (embedder, reviewer)
    are injected fakes — exactly how you'd test your own extraction code.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The payload model

    With the real [`StructuredExtractor`][ragdoc.extraction.StructuredExtractor], **your
    Pydantic model is the schema**: its docstring and `Field` descriptions are serialized
    into the JSON schema the LLM is constrained to. The stub below extracts the same model.
    """)
    return


@app.cell
def _():
    from pydantic import BaseModel, Field

    class Person(BaseModel):
        """A person mentioned in the text."""

        name: str = Field(description="Full name as written in the text.")
        role: str | None = Field(default=None, description="Role or title, when stated.")

    return (BaseModel, Field, Person)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A sample corpus

    Two small HTML files in a temporary directory. The second file mentions Ada Lovelace
    again — under a different role, so the two occurrences are *distinct mentions* that
    resolution will later merge into *one entity*.
    """)
    return


@app.cell
def _():
    import tempfile
    from pathlib import Path

    corpus_dir = Path(tempfile.mkdtemp(prefix="ragdoc-extraction-"))

    (corpus_dir / "notes.html").write_text(
        "<html><body>"
        "<h1>Analytical Engine notes</h1>"
        "<p>Ada Lovelace, a mathematician, annotated the engine design.</p>"
        "<p>Charles Babbage, a designer, built the engine itself.</p>"
        "</body></html>",
        encoding="utf-8",
    )
    (corpus_dir / "letters.html").write_text(
        "<html><body>"
        "<h1>Correspondence</h1>"
        "<p>Ada Lovelace, a countess, corresponded about programming the engine.</p>"
        "</body></html>",
        encoding="utf-8",
    )

    paths = sorted(corpus_dir.glob("*.html"))
    print(f"corpus: {[p.name for p in paths]}")
    return (Path, corpus_dir, paths)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## A stub `Extractor`

    Anything with `async extract(document) -> list[Mention[PayloadT]]` satisfies the
    [`Extractor`][ragdoc.extraction.Extractor] protocol. The contract: extraction is a
    **pure query** — it never reads or writes `document.metadata` keys of its own (each
    mention's locator *copies* the document metadata).

    For real LLM extraction, swap this stub for
    `StructuredExtractor(Person)` — everything downstream stays identical.
    """)
    return


@app.cell
def _(Person):
    from ragdoc.document import Document
    from ragdoc.extraction import Mention, mint_mention_id

    # name -> role per source file, keyed on phrasing found in the corpus above
    KNOWN_PEOPLE: list[tuple[str, str]] = [
        ("Ada Lovelace", "mathematician"),
        ("Charles Babbage", "designer"),
        ("Ada Lovelace", "countess"),
    ]

    class StubPersonExtractor:
        """Deterministic Extractor: matches known (name, role) phrases — no LLM, no network."""

        payload_model = Person

        async def extract(self, document: Document) -> list[Mention[Person]]:
            text = " ".join(el.text for el in document.elements)
            source_id = document.source_id or document.id
            mentions: list[Mention[Person]] = []
            for name, role in KNOWN_PEOPLE:
                if f"{name}, a {role}" not in text:
                    continue
                ordinal = len(mentions)
                payload = Person(name=name, role=role)
                mentions.append(
                    Mention(
                        mention_id=mint_mention_id(source_id, None, ordinal, payload),
                        source_id=source_id,
                        source_hash=document.source_hash or "",
                        ordinal=ordinal,
                        metadata=dict(document.metadata),
                        payload=payload,
                    )
                )
            return mentions

    extractor = StubPersonExtractor()
    return (Document, KNOWN_PEOPLE, Mention, StubPersonExtractor, extractor, mint_mention_id)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Layer 1 — syncing mentions with `MentionStorePipeline`

    The pipeline composes an `IngestPipeline` (parse → process), a splitter (extraction
    runs per context-sized split; here a pass-through since the documents are tiny), the
    extractor, and a [`MentionStore`][ragdoc.extraction.MentionStore] — with per-source
    change detection: an **unchanged file is skipped with no extraction call**.
    """)
    return


@app.cell
async def _(Person, corpus_dir, extractor, paths):
    from ragdoc.extraction import LocalMentionStore, MentionStorePipeline
    from ragdoc.pipeline import IngestPipeline

    mention_store = LocalMentionStore(corpus_dir / "mentions", Person)

    mention_pipeline = MentionStorePipeline(
        ingest=IngestPipeline(),
        extractor=extractor,
        mention_store=mention_store,
        splitter=lambda document: [document],  # tiny docs: skip token splitting
    )

    result_first = await mention_pipeline.run(paths)
    print(f"first run:  processed={sorted(result_first.processed)} skipped={result_first.skipped}")
    return (IngestPipeline, LocalMentionStore, MentionStorePipeline, mention_pipeline, mention_store, result_first)


@app.cell
async def _(mention_pipeline, paths):
    # Re-running is cheap: unchanged files are detected by their byte hash and skipped.
    result_second = await mention_pipeline.run(paths)
    print(f"second run: processed={result_second.processed} skipped={sorted(result_second.skipped)}")
    return (result_second,)


@app.cell
async def _(mention_store):
    mentions = await mention_store.list_mentions()
    for _m in sorted(mentions, key=lambda m: (m.source_id, m.ordinal)):
        print(f"{_m.source_id}  #{_m.ordinal}  {_m.payload.name} ({_m.payload.role})")
    return (mentions,)


@app.cell
def test_mention_sync(mentions, result_first, result_second):
    # Both files produced mentions on the first run…
    assert sorted(result_first.processed) == ["letters.html", "notes.html"]
    assert result_first.skipped == []
    # …and were skipped (no extraction) on the unchanged second run.
    assert result_second.processed == []
    assert sorted(result_second.skipped) == ["letters.html", "notes.html"]
    # 3 provenance-tagged mentions: 2 in notes.html, 1 in letters.html.
    assert len(mentions) == 3
    assert all(m.source_hash for m in mentions)
    assert all(m.metadata["filename"] == m.source_id for m in mentions)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Layer 2 — resolving mentions into entities

    Three mentions, but only two real people. `EntityResolutionPipeline` clusters the
    mention store: exact-identity seeding, embedding-based blocking, LLM review, merge.
    Both collaborators are **injected**, so fakes make the loop fully deterministic:

    - `embed` — any `async (list[str]) -> list[list[float]]`. Our fake maps Ada-texts and
      Babbage-texts to orthogonal vectors, so only the two Ada clusters look similar.
    - `reviewer` — any `async (list[str]) -> ReviewResult`. Our fake confirms every
      candidate group (production uses
      [`make_llm_reviewer`][ragdoc.extraction.make_llm_reviewer]).
    """)
    return


@app.cell
def _():
    from ragdoc.extraction import ReviewGroup, ReviewResult

    async def embed_fn(texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "Ada" in t else [0.0, 1.0] for t in texts]

    async def reviewer_fn(texts: list[str]) -> ReviewResult:
        return ReviewResult(groups=[ReviewGroup(members=list(range(len(texts))), confidence=1.0)])

    return (ReviewGroup, ReviewResult, embed_fn, reviewer_fn)


@app.cell
async def _(Person, corpus_dir, embed_fn, mention_store, reviewer_fn):
    from ragdoc.extraction import EntityResolutionPipeline, LocalEntityStore

    entity_store = LocalEntityStore(corpus_dir / "entities", Person)

    resolution = EntityResolutionPipeline(
        mention_store=mention_store,
        entity_store=entity_store,
        embed=embed_fn,
        reviewer=reviewer_fn,
    )
    resolution_result = await resolution.resolve()

    entities = sorted(resolution_result.entities, key=lambda e: e.payload.name)
    for _e in entities:
        print(f"{_e.payload.name}: {len(_e.member_mention_ids)} mention(s) from {sorted(_e.source_ids)}")
    return (EntityResolutionPipeline, LocalEntityStore, entities, entity_store, resolution, resolution_result)


@app.cell
def test_resolution(entities, resolution_result):
    # 3 mentions resolved into 2 canonical entities.
    assert len(entities) == 2
    ada, babbage = entities
    # Ada's two differently-phrased mentions merged into one entity spanning both files.
    assert ada.payload.name == "Ada Lovelace"
    assert len(ada.member_mention_ids) == 2
    assert sorted(ada.source_ids) == ["letters.html", "notes.html"]
    # Babbage stayed a single-mention entity.
    assert babbage.payload.name == "Charles Babbage"
    assert len(babbage.member_mention_ids) == 1
    # The confident fake reviewer left nothing pending.
    assert resolution_result.pending == []
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Going further

    - **Real LLM extraction** — replace the stub with
      `StructuredExtractor(Person)` (requires `ragdoc[extraction,llm]` and
      `configure(openai_client=AsyncOpenAI())`); nothing else changes.
    - **Typed knowledge graphs** — declare node/edge models and legal patterns in a
      [`GraphSchema`][ragdoc.extraction.GraphSchema], extract with
      `KnowledgeGraphExtractor(schema)` through the *same* `MentionStorePipeline`, then
      resolve with `KnowledgeGraphResolutionPipeline`.
    - **Qdrant-backed stores** — swap the `Local*` stores for `QdrantMentionStore` /
      `QdrantEntityStore` / `QdrantGraphStore` (`ragdoc[qdrant]`).
    - **Review before writing** — `mention_pipeline.plan(paths)` returns a serializable
      `ChangeSet[Mention[Person]]`; see the [Sync Engine guide](../guide/sync-engine.md).

    ## See Also

    - [Extraction Guide](../guide/extraction.md) — the full three-layer workflow
    - [API Reference: Extraction](../api/extraction.md)
    """)
    return


if __name__ == "__main__":
    app.run()
