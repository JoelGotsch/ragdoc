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

Every LLM call in the library goes through one shared layer: structural client protocols,
one client-resolution policy, and one retry policy. `RagdocConfig.openai_client` holds an
`LLMClient` (the `ChatClient` × `EmbeddingsClient` intersection). See the
[LLM API page](llm.md) for the full reference.
