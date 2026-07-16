# LLM API

The shared LLM reliability layer (`ragdoc.llm`). Every LLM call in the library goes through
this one layer: structural client protocols (`ChatClient`, `EmbeddingsClient`, and their
intersection `LLMClient` — what
[`RagdocConfig.openai_client`](config.md#ragdocconfig) holds), one client-resolution
policy, and one retry policy on the non-beta `client.chat.completions.parse` namespace.

## Client protocols

### LLMClient

::: ragdoc.llm.LLMClient
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

### ChatClient

::: ragdoc.llm.ChatClient
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

### EmbeddingsClient

::: ragdoc.llm.EmbeddingsClient
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

## Client resolution and calls

### resolve_openai_client

::: ragdoc.llm.resolve_openai_client
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### call_structured

::: ragdoc.llm.call_structured
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### retry_llm

::: ragdoc.llm.retry_llm
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

## Errors

### LLMError

::: ragdoc.llm.LLMError
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

### LLMRefusalError

::: ragdoc.llm.LLMRefusalError
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4

### LLMNotConfiguredError

::: ragdoc.llm.LLMNotConfiguredError
    options:
      show_root_heading: true
      show_source: false
      heading_level: 4
