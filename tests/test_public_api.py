"""Public API surface — the intended facade, exactly."""

import pytest

import ragdoc

EXPECTED_ALL = {
    # core model
    "Document",
    "BaseElement",
    "Heading",
    "Paragraph",
    "Table",
    "Image",
    "DocumentList",
    "Footnote",
    "RawText",
    "ExternalRef",
    "InlineRef",
    # metadata typing
    "BaseMetadata",
    # entry points
    "load",
    "DocumentPipeline",
    "IngestPipeline",
    "ChunkPipeline",
    "DocumentStorePipeline",
    "VectorStorePipeline",
    # stages
    "SimpleChunker",
    "LLMChunker",
    "TokenSplitter",
    "split_document",
    # outputs & sync
    "Chunk",
    "ChangeSet",
    "UpdateResult",
    # config + LLM client typing
    "configure",
    "RagdocConfig",
    "ChatClient",
    "EmbeddingsClient",
    "LLMClient",
}


def test_all_is_exactly_the_facade():
    assert set(ragdoc.__all__) == EXPECTED_ALL


def test_every_name_importable():
    for name in ragdoc.__all__:
        assert getattr(ragdoc, name) is not None


@pytest.mark.parametrize(
    "legacy",
    [
        "from_path",
        "load_document",
        "DocumentSource",
        "HTMLFile",
        "ExcelPackage",
        "WordFile",
        "AzureJSONFile",
        "AzureAnalyzeRun",
        "PandocFile",
        "HTMLSource",
        "ExcelSource",
    ],
)
def test_legacy_names_gone_from_root(legacy):
    assert not hasattr(ragdoc, legacy)


def test_autoparser_module_deleted():
    with pytest.raises(ModuleNotFoundError):
        import ragdoc.pipeline.parser  # noqa: F401  # pyright: ignore[reportMissingImports]


def test_llama_index_integration_deleted():
    with pytest.raises(ModuleNotFoundError):
        import ragdoc.integrations.llama_index  # noqa: F401  # pyright: ignore[reportMissingImports]
