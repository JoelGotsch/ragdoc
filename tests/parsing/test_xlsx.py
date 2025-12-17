from io import StringIO
from pathlib import Path

import pandas as pd

from ragdoc.parsing import ExcelConfig
from ragdoc.parsing.xlsx import load_excel


def test_xlsx(xlsx_file_path: Path):
    config = ExcelConfig(
        sheet_params={
            "Sheet1": {},
            "Example": {"skiprows": 3},
        }
    )
    document = load_excel(xlsx_file_path, config)
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
    document = load_excel(xlsx_file_path, config)
    df = pd.read_html(StringIO(document.tables[0].html), header=0, index_col=0)[0]
    assert df.columns.tolist() == ["a", "b", "c"]
    assert df.index.tolist() == ["row1", "row2", "row3"]


def test_xlsx_multiline(xlsx_file_path: Path):
    config = ExcelConfig(
        sheet_params={
            "multiline_header": {"header": [0, 1], "index_col": 0},
        }
    )
    document = load_excel(xlsx_file_path, config)
    df = pd.read_html(StringIO(document.tables[0].html), header=[0, 1], index_col=0)[0]
    assert df.index.tolist() == ["row1", "row2"]
    assert df.columns.tolist() == [
        ("this is a multiline header", "col1"),
        ("this is a multiline header", "col2"),
        ("this is a multiline header", "col3"),
    ]
    assert df.loc["row1", ("this is a multiline header", "col1")] == "value"


def test_xlsx_load_all(xlsx_file_path: Path):
    document = load_excel(xlsx_file_path)
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
    document = load_excel(xlsx_file_path)
    assert document.metadata["filename"] == "test.xlsx"
    assert Path(document.source_path).match("*/tests/data/test.xlsx")


def test_sheet_params_override_truthy_default_params(monkeypatch):
    """Per-sheet params must merge over (and win against) truthy default_params (Phase 0, bug 1)."""
    from types import SimpleNamespace

    from ragdoc.parsing.xlsx.load import generate_document as xlsx_generate_document

    captured: dict = {}

    def fake_read_excel(excel_file, sheet_name, **params):
        captured[sheet_name] = params
        return pd.DataFrame({"a": [1]})

    monkeypatch.setattr("ragdoc.parsing.xlsx.load.pd.read_excel", fake_read_excel)
    fake_file = SimpleNamespace(sheet_names=["Sheet1"])
    config = ExcelConfig(default_params={"skiprows": 1}, sheet_params={"Sheet1": {"skiprows": 5, "nrows": 2}})
    xlsx_generate_document(fake_file, config)  # type: ignore[arg-type]
    assert captured["Sheet1"] == {"skiprows": 5, "nrows": 2}
