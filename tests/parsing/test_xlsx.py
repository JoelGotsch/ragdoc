from io import StringIO
from pathlib import Path

import pandas as pd

from ragdoc.parsing import ExcelConfig, ExcelPackage, load_file


def test_xlsx(xlsx_file_path: Path):
    config = ExcelConfig(
        sheet_params={
            "Sheet1": {},
            "Example": {"skiprows": 3},
        }
    )
    excel_file = ExcelPackage(file_path=xlsx_file_path, config=config)
    document = load_file(excel_file)
    assert len(document.headings) == 2
    assert [h.text for h in document.headings] == [
        "Sheet1",
        "Example",
    ]
    assert len(document.tables) == 2
    assert document.tables[0].html == document.tables[1].html


def test_xlsx_heading(xlsx_file_path: Path):
    config = ExcelConfig(
        sheet_params={
            "Example": {"header": 3, "index_col": 0},
        }
    )
    excel_file = ExcelPackage(file_path=xlsx_file_path, config=config)
    document = load_file(excel_file)
    df = pd.read_html(StringIO(document.tables[0].html), header=0, index_col=0)[0]
    assert df.columns.tolist() == ["a", "b", "c"]
    assert df.index.tolist() == ["row1", "row2", "row3"]


def test_xlsx_multiline(xlsx_file_path: Path):
    config = ExcelConfig(
        sheet_params={
            "multiline_header": {"header": [0, 1], "index_col": 0},
        }
    )
    excel_file = ExcelPackage(file_path=xlsx_file_path, config=config)
    document = load_file(excel_file)
    df = pd.read_html(StringIO(document.tables[0].html), header=[0, 1], index_col=0)[0]
    assert df.index.tolist() == ["row1", "row2"]
    assert df.columns.tolist() == [
        ("this is a multiline header", "col1"),
        ("this is a multiline header", "col2"),
        ("this is a multiline header", "col3"),
    ]
    assert df.loc["row1", ("this is a multiline header", "col1")] == "value"


def test_xlsx_load_all(xlsx_file_path: Path):
    excel_file = ExcelPackage(file_path=xlsx_file_path)
    document = load_file(excel_file)
    assert len(document.tables) == 4
    assert len(document.headings) == 4
    document_titles = [h.text for h in document.headings]
    assert document_titles == [
        "Sheet1",
        "Sheet2",
        "Example",
        "multiline_header",
    ]


def test_xlsx_metadata(xlsx_file_path: Path):
    """Test document source fields are correctly set."""
    excel_file = ExcelPackage(file_path=xlsx_file_path)
    document = load_file(excel_file)
    assert document.metadata["filename"] == "test.xlsx"
    assert Path(document.source_path).match("*/tests/data/test.xlsx")
