import json  # type: ignore

from ragdoc.parsing.base import load_file
from ragdoc.document import Document, join_documents

from pydantic import BaseModel, Field
from pathlib import Path

from ragdoc.parsing.parser import Parser

try:
    from textract_preprocessing.textract_caller import TextractCaller
    textract_available = True
except ImportError:
    textract_available = False

class PDFFile(BaseModel):
    file_path: str | Path = Field(
        description="The path to the file. Can be local path or S3 path to file. If this is a s3 path, then s3 path must be left empty."
    )
    textract_output_path: str | None = Field(default=None, description="The path to save the textract output to")
    s3_path: str | None = Field(default=None, description="The path to the S3 bucket. Only used if file_path is a local path.")

    @property
    def s3_file_path(self) -> str | None:
        if str(self.file_path).startswith("s3://"):
            return None
        assert self.s3_path, "If file_path is a local path, s3_path must be provided"
        file_path = Path(self.file_path)
        assert self.s3_path.endswith("/")
        return self.s3_path + file_path.name


class TextractJSONFile(BaseModel):
    file_path: str | Path
    pdf_path: str | None = Field(default=None, description="Path to the original PDF file")


@load_file.register
def load_pdf(file_obj: PDFFile) -> Document:
    """Parse a PDF via AWS Textract into a Document. Sets ``source_path`` and ``metadata["filename"]``."""
    if not textract_available:
        raise ImportError("Please install extra 'textract' to use this function")
    textract_caller = TextractCaller(region="eu-central-1")
    file_path = Path(file_obj.file_path)
    response = textract_caller(str(file_path), file_obj.s3_file_path)
    if file_obj.textract_output_path:
        output_path = Path(file_obj.textract_output_path)
        if output_path.is_dir():
            output_path = output_path / f"{file_path.stem}.json"
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(response, fh)

    textract = Textract.from_list(l=response, file_path=str(file_path), pdf_path=str(file_path))
    documents, orphans = generate_textract_documents(textract)
    doc = join_documents([Document(elements=orphans, source_path=str(file_path), metadata={"filename": file_path.name})] + documents)
    doc.parser = "textract"
    return doc


@load_file.register
def load_textract_json(file_obj: TextractJSONFile) -> Document:
    """Parse a Textract JSON output file into a Document. Sets ``source_path`` and ``metadata["filename"]``."""
    file_path = Path(file_obj.file_path)
    textract = Textract.from_json(file_path=file_obj.file_path, pdf_path=file_obj.pdf_path)
    documents, orphans = generate_textract_documents(textract)
    doc = join_documents([Document(elements=orphans, source_path=str(file_path), metadata={"filename": file_path.name})] + documents)
    doc.parser = "textract"
    return doc


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

def parse_textract_json_file(path: Path) -> Document:
    """Parse a Textract JSON file (standalone function for reuse/testing)."""
    return load_textract_json(TextractJSONFile(file_path=path))


class TextractJSONParser(Parser):
    name: str = "textract"
    patterns: list[str] = [".textract.json"]
    description: str = "AWS Textract JSON result files"

    async def __call__(self, path: Path) -> Document:
        return parse_textract_json_file(path)


def _register() -> None:
    from ragdoc.parsing.registry import register_parser

    register_parser(TextractJSONParser())
