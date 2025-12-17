# Document API

Core data models for the ragdoc pipeline. The `Document` model is the single source of truth
throughout all pipeline stages.

## Document

::: ragdoc.document.Document
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## BaseElement

::: ragdoc.document.BaseElement
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Element Types

### Heading

::: ragdoc.document.Heading
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### Paragraph

::: ragdoc.document.Paragraph
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### Table

::: ragdoc.document.Table
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### Image

::: ragdoc.document.Image
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### DocumentList

::: ragdoc.document.DocumentList
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### Footnote

::: ragdoc.document.Footnote
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### RawText

::: ragdoc.document.RawText
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

## Reference Models

### InlineRef

::: ragdoc.document.InlineRef
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### ExternalRef

::: ragdoc.document.ExternalRef
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

## Enums

### ElementTypeEnum

::: ragdoc.document.ElementTypeEnum
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

## Utility Functions

### concat_documents

::: ragdoc.document.concat_documents
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### join_documents

::: ragdoc.document.join_documents
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### image_fields_from_html

::: ragdoc.document.image_fields_from_html
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### footnote_fields_from_html

::: ragdoc.document.footnote_fields_from_html
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4
