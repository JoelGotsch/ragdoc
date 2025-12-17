# Configuration API

Global configuration for ragdoc, including LLM client settings and Azure credentials.

## RagdocConfig

::: ragdoc.config.RagdocConfig
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## configure

::: ragdoc.config.configure
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## get_config

::: ragdoc.config.get_config
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## LLM reliability layer (`ragdoc.llm`)

Every LLM call in the library goes through one shared layer: structural client protocols
(`ChatClient`, `EmbeddingsClient`, and their intersection `LLMClient` — what
`RagdocConfig.openai_client` holds), one client-resolution policy, and one retry policy on the
non-beta `client.chat.completions.parse` namespace.

::: ragdoc.llm.LLMClient
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

::: ragdoc.llm.ChatClient
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

::: ragdoc.llm.EmbeddingsClient
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

::: ragdoc.llm.resolve_openai_client
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

::: ragdoc.llm.call_structured
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

::: ragdoc.llm.retry_llm
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

::: ragdoc.llm.LLMRefusalError
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

::: ragdoc.llm.LLMNotConfiguredError
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3
