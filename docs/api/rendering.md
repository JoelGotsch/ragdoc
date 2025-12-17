# Rendering API

Classes and functions for converting `Document` objects into formatted text strings.

## Renderer

::: ragdoc.rendering.base.Renderer
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## RenderContext

::: ragdoc.rendering.base.RenderContext
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## OutputFormat

::: ragdoc.rendering.base.OutputFormat
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## ExternalRefProvider

::: ragdoc.rendering.base.ExternalRefProvider
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## DocumentMetadata

::: ragdoc.rendering.base.DocumentMetadata
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## MetadataValue

::: ragdoc.metadata.MetadataValue
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Element Renderers

Element renderers are `singledispatch` functions. They always return HTML. The `Renderer`
class handles format conversion (HTML → Markdown, plain text, etc.).

### render_for_prompt

::: ragdoc.rendering.elements.render_for_prompt
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### render_raw

::: ragdoc.rendering.elements.render_raw
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4
