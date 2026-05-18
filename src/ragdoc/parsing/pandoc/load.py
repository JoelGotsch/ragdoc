import os

from pypandoc import convert_file

from ragdoc.parsing.html.load import generate_document as html_generate_document, HTML
from ragdoc.document import Paragraph, Document, Table

class PandocHTML(HTML):
    @classmethod
    def from_file(cls, path: str) -> str:

        _, ext = os.path.splitext(path)

        content = convert_file(
            source_file=path,
            format=ext.lower().replace(".", ""),
            to="html",
            extra_args=("--embed-resources",)
        )

        return cls(content=content)


def generate_document(html: PandocHTML) -> Document:
    document = html_generate_document(html=html)
    document.parser = "pandoc"
    return document
