# LLM API

The shared LLM reliability layer (`ragdoc.llm`). Every LLM call in the library goes through
this one layer: structural client protocols (`ChatClient`, `EmbeddingsClient`, and their
intersection `LLMClient`), one client-resolution policy, and one retry policy on the non-beta
`client.chat.completions.parse` namespace.
[`RagdocConfig.openai_client`](config.md#ragdocconfig) is typed
`ChatClient | EmbeddingsClient | None`: it accepts a client with either capability, and each
consumer demands the one it uses.

The retry policy in short: 429, connection/timeout errors and 5xx are retried with jittered
exponential backoff; a `Retry-After` header is honoured only up to `backoff_max`; other 4xx,
refusals and validation errors are not retried. For a real `openai.AsyncOpenAI` the library
disables the SDK's own retries per call (`without_sdk_retries`), so there is exactly one retry
layer. The module docstring of `ragdoc.llm` has the full statement, including timeouts.

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

### without_sdk_retries

::: ragdoc.llm.without_sdk_retries
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

## Failure classification

Shared predicates for call sites that degrade per item or shrink their input — use these
instead of matching exception types locally.

### is_llm_failure

::: ragdoc.llm.is_llm_failure
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### is_size_related_failure

::: ragdoc.llm.is_size_related_failure
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
