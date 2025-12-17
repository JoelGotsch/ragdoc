# TODO: Long-term, each integration (llama_index, langchain, etc.) should live in its own
# separate repository as an optional add-on package. For now they live here.
#
# To add a new framework integration, create a sibling folder (e.g. integrations/langchain/)
# and follow the same pattern: guard the import with a try/except ImportError so the rest
# of ragdoc works without the extra dependency installed.

try:
    import llama_index  # type: ignore[reportMissingImports]  # optional, untyped integration dependency

    _llama_index_available = True
except ImportError:
    _llama_index_available = False

from collections.abc import Callable
from copy import deepcopy

from ragdoc.chunking import Chunk


def node_dict_to_document_fragment(node_dict: dict, content_replacement_key: str = "__prompt_content_key__") -> Chunk:
    """Transforms a dict of a `llama_index.core.schema.TextNode` to a Chunk.

    Note: `llama_index.core.schema.TextNode` doesnt differentiate between embedding-content and prompt-content.
    Therefore, if they differ, the prompt content is stored in the metadata under the key specified by content_replacement_key.

    Args:
        node_dict (dict): dict of a `llama_index.core.schema.TextNode`
        content_replacement_key (str, optional): The key used to store the prompt-content in the metadata. Defaults to "__prompt_content_key__".

    Returns:
        Chunk: The transformed chunk
    """
    embedding_content = node_dict["text"]
    metadata = deepcopy(node_dict.get("metadata", {}))

    if content_replacement_key not in metadata:
        prompt_content = node_dict["text"]
    else:
        prompt_content = metadata[content_replacement_key]
        del metadata[content_replacement_key]

    return Chunk(
        id=node_dict["id_"],
        embedding_content=embedding_content,
        prompt_content=prompt_content,
        metadata=metadata,
    )


def document_fragment_to_node_dict(fragment: Chunk, content_replacement_key: str = "__prompt_content_key__") -> dict:
    """Transforms a Chunk to a dict of a `llama_index.core.schema.TextNode`.

    Note: `llama_index.core.schema.TextNode` doesnt differentiate between embedding-content and prompt-content.
    Therefore, if they differ, the prompt content is stored in the metadata under the key specified by content_replacement_key.

    Args:
        chunk (Chunk): The chunk to be transformed
        content_replacement_key (str, optional): The key used to store the prompt-content in the metadata. Defaults to "__prompt_content_key__".

    Returns:
        dict: The dict of a `llama_index.core.schema.TextNode`
    """
    metadata = deepcopy(fragment.metadata)
    text = fragment.embedding_content
    if content_replacement_key not in metadata:
        metadata[content_replacement_key] = fragment.prompt_content

    # Chunk no longer stores per-key embed/prompt inclusion lists (removed in the
    # chunking refactor), so include all metadata in both representations and
    # exclude only the internal key that carries the prompt content.
    excluded_embed_metadata_keys = [content_replacement_key]
    excluded_llm_metadata_keys = [content_replacement_key]
    return dict(
        id_=fragment.id,
        text=text,
        metadata=metadata,
        embedding=None,
        excluded_embed_metadata_keys=excluded_embed_metadata_keys,
        excluded_llm_metadata_keys=excluded_llm_metadata_keys,
    )
