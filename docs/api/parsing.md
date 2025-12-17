# Parsing API

Functions and source types for loading documents from files.

## load

::: ragdoc.parsing.load
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Parser

::: ragdoc.parsing.parser.Parser
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## register_parser

::: ragdoc.parsing.register_parser
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## get_parser

::: ragdoc.parsing.get_parser
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## get_registered_parsers

::: ragdoc.parsing.get_registered_parsers
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## unregister_parser

::: ragdoc.parsing.unregister_parser
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## ParserRegistration

::: ragdoc.parsing.ParserRegistration
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## MultiSourceParser

::: ragdoc.parsing.multi_source.MultiSourceParser
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## SecondaryResolver

::: ragdoc.parsing.resolvers.SecondaryResolver
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## sibling_resolver

::: ragdoc.parsing.resolvers.sibling_resolver
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

---

## MinerU Parser

See the [MinerU guide](../guide/parsing.md#mineru-_middle-json) for an overview of
handler swapping, image loading, and metadata propagation.

### MinerUParser

::: ragdoc.parsing.mineru.MinerUParser
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### MinerUExtractor

::: ragdoc.parsing.mineru.MinerUExtractor
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### parse_mineru_file

::: ragdoc.parsing.mineru.parse_mineru_file
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### CoreExtractionMiddleware

::: ragdoc.parsing.mineru.CoreExtractionMiddleware
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### ExtractionConfig

::: ragdoc.parsing.mineru.ExtractionConfig
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### Handler functions

#### handle_title_block

::: ragdoc.parsing.mineru.handle_title_block
    options:
      show_root_heading: true
      show_source: true
      heading_level: 5

#### handle_text_block

::: ragdoc.parsing.mineru.handle_text_block
    options:
      show_root_heading: true
      show_source: true
      heading_level: 5

#### handle_list_block

::: ragdoc.parsing.mineru.handle_list_block
    options:
      show_root_heading: true
      show_source: true
      heading_level: 5

#### handle_code_block

::: ragdoc.parsing.mineru.handle_code_block
    options:
      show_root_heading: true
      show_source: true
      heading_level: 5

#### handle_table_block

::: ragdoc.parsing.mineru.handle_table_block
    options:
      show_root_heading: true
      show_source: true
      heading_level: 5

#### handle_image_block

::: ragdoc.parsing.mineru.handle_image_block
    options:
      show_root_heading: true
      show_source: true
      heading_level: 5

#### handle_discarded_as_metadata

::: ragdoc.parsing.mineru.handle_discarded_as_metadata
    options:
      show_root_heading: true
      show_source: true
      heading_level: 5

#### handle_discarded_as_footnote

::: ragdoc.parsing.mineru.handle_discarded_as_footnote
    options:
      show_root_heading: true
      show_source: true
      heading_level: 5

#### handle_discarded_as_raw_text

::: ragdoc.parsing.mineru.handle_discarded_as_raw_text
    options:
      show_root_heading: true
      show_source: true
      heading_level: 5

#### handle_discarded_drop

::: ragdoc.parsing.mineru.handle_discarded_drop
    options:
      show_root_heading: true
      show_source: true
      heading_level: 5

### HTML builder helpers

#### build_heading_html

::: ragdoc.parsing.mineru.build_heading_html
    options:
      show_root_heading: true
      show_source: true
      heading_level: 5

#### build_text_html

::: ragdoc.parsing.mineru.build_text_html
    options:
      show_root_heading: true
      show_source: true
      heading_level: 5

!!! note "MinerUMiddleDocument is internal"
    `MinerUMiddleDocument` is an internal type produced by the MinerU library.
    It is not part of the public API and is not documented here.
    Use `MinerUParser`, `parse_mineru_file`, or `MinerUExtractor` instead.

---

## RagdocJson Parser

### RagdocJsonParser

::: ragdoc.parsing.ragdoc_json.RagdocJsonParser
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### ProvenanceMode

::: ragdoc.parsing.ragdoc_json.ProvenanceMode
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

### parse_ragdoc_json

::: ragdoc.parsing.ragdoc_json.parse_ragdoc_json
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

---

## Parser Configuration

### ExcelConfig

::: ragdoc.parsing.xlsx.ExcelConfig
    options:
      show_root_heading: true
      show_source: true
      heading_level: 4

