from pathlib import Path

from ragdoc.parsing import load_file, AzureJSONFile


def test_load_json_file(azure_test_pdf_json_file_path: Path, data_path: Path):
    # Test loading azure json file without adding a source pdf
    document = load_file(AzureJSONFile(file_path=azure_test_pdf_json_file_path))
    
    assert document.name == "New super amazing Document"
    assert len(document.paragraphs) == 18
    assert len(document.tables) == 1
    assert len(document.images) == 0
    
    # Test loading azure json file with adding a source pdf
    document = load_file(AzureJSONFile(
        file_path=azure_test_pdf_json_file_path,
        source_path=data_path / "test.pdf"
    ))
    
    assert document.name == 'New super amazing Document'
    assert len(document.headings) == 5
    assert len(document.paragraphs) == 18
    assert len(document.tables) == 1
    assert len(document.images) == 1

    # The image has the content of the figure tag stored in "alt"
    assert document.images[0].alt == """
WHEN YOU FIND A DANK MEME
TO SEND TO YOUR FRIENDS
"""

def test_load_json_file_no_figures(azure_test_pdf_json_file_path_no_figures: Path, data_path: Path):
    # Test loading azure json file with adding a source pdf
    document = load_file(AzureJSONFile(
        file_path=azure_test_pdf_json_file_path_no_figures,
        source_path=data_path / "test.pdf"
    ))
    
    assert document.name == 'New super amazing Document'
    assert len(document.headings) == 5
    assert len(document.paragraphs) == 18
    assert len(document.tables) == 1
    assert len(document.images) == 0

def test_recovered_heading_information(azure_test_pdf_json_file_path: Path, data_path: Path):
    # Test loading azure json file with adding a source pdf
    document = load_file(AzureJSONFile(
        file_path=azure_test_pdf_json_file_path,
        source_path=data_path / "test.pdf"
    ))

    headings = [h.text for h in document.headings]
    assert headings ==   [
        'New super amazing Document',
        'Second layer',
        'Second second layer',
        'Third second layer',
        'Numbered items',
    ]

    assert [h.level for h in document.headings] == [1, 2, 2, 3, 1]


def test_azure_json_metadata(azure_test_pdf_json_file_path: Path):
    """Test document source fields are correctly set from AzureJSONFile."""
    document = load_file(AzureJSONFile(file_path=azure_test_pdf_json_file_path))
    assert document.metadata["filename"] == "test_pdf_azure_di.json"
    assert Path(document.source_path).match("*/tests/data/test_pdf_azure_di.json")


def test_azure_analyze_run_metadata(azure_test_pdf_json_file_path: Path, data_path: Path):
    """Test document source fields use source_path when provided."""
    import json
    from ragdoc.parsing import load_file
    from ragdoc.parsing.azure_di import AzureAnalyzeRun

    with open(azure_test_pdf_json_file_path) as fh:
        analyze_dict = json.load(fh)
    source = data_path / "test.pdf"
    document = load_file(AzureAnalyzeRun(analyze_dict=analyze_dict, source_path=source))
    assert document.metadata["filename"] == "test.pdf"
    assert Path(document.source_path).match("*/tests/data/test.pdf")
