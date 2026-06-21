"""Tests for src/ragdoc/cli.py Click commands."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

from click.testing import CliRunner

from ragdoc.cli import document_processing
from ragdoc.document import Document


def _empty_doc() -> Document:
    return Document(elements=[])


# ---------------------------------------------------------------------------
# parse command
# ---------------------------------------------------------------------------


class TestParseCommand:
    def test_parse_creates_output_json(self, tmp_path: Path, html_file_path: Path):
        runner = CliRunner()
        output_dir = tmp_path / "out"
        with patch("ragdoc.cli.load", new_callable=AsyncMock) as mock_load:
            mock_load.return_value = _empty_doc()
            result = runner.invoke(
                document_processing,
                ["parse", str(html_file_path), "--output", str(output_dir)],
            )

        assert result.exit_code == 0
        assert "Processing" in result.output
        assert output_dir.exists()
        json_files = list(output_dir.glob("*.document.json"))
        assert len(json_files) == 1

    def test_parse_value_error_printed_and_continues(self, tmp_path: Path, html_file_path: Path):
        runner = CliRunner()
        output_dir = tmp_path / "out"
        with patch("ragdoc.cli.load", new_callable=AsyncMock) as mock_load:
            mock_load.side_effect = ValueError("unsupported format")
            result = runner.invoke(
                document_processing,
                ["parse", str(html_file_path), "--output", str(output_dir)],
            )

        assert result.exit_code == 0
        assert "Error: unsupported format" in result.output

    def test_parse_runtime_error_printed_and_continues(self, tmp_path: Path, html_file_path: Path):
        runner = CliRunner()
        output_dir = tmp_path / "out"
        with patch("ragdoc.cli.load", new_callable=AsyncMock) as mock_load:
            mock_load.side_effect = RuntimeError("connection failed")
            result = runner.invoke(
                document_processing,
                ["parse", str(html_file_path), "--output", str(output_dir)],
            )

        assert result.exit_code == 0
        assert "Error: connection failed" in result.output

    def test_parse_no_files_succeeds(self, tmp_path: Path):
        runner = CliRunner()
        result = runner.invoke(
            document_processing,
            ["parse", "--output", str(tmp_path / "out")],
        )
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# chunk command
# ---------------------------------------------------------------------------


class TestChunkCommand:
    def test_chunk_creates_rendered_output(self, tmp_path: Path, html_file_path: Path):
        from ragdoc.document import Paragraph

        doc = Document(elements=[Paragraph(html_content="<p>Hello world</p>")])
        runner = CliRunner()
        output_dir = tmp_path / "out"
        with patch("ragdoc.cli.load", new_callable=AsyncMock) as mock_load:
            mock_load.return_value = doc
            result = runner.invoke(
                document_processing,
                ["chunk", str(html_file_path), "--output", str(output_dir), "--format", "md"],
            )

        assert result.exit_code == 0
        assert "Processing" in result.output
        md_files = list(output_dir.glob("*.md"))
        assert len(md_files) == 1

    def test_chunk_value_error_printed_and_continues(self, tmp_path: Path, html_file_path: Path):
        runner = CliRunner()
        output_dir = tmp_path / "out"
        with patch("ragdoc.cli.load", new_callable=AsyncMock) as mock_load:
            mock_load.side_effect = ValueError("parse failed")
            result = runner.invoke(
                document_processing,
                ["chunk", str(html_file_path), "--output", str(output_dir)],
            )

        assert result.exit_code == 0
        assert "Error: parse failed" in result.output

    def test_chunk_format_option(self, tmp_path: Path, html_file_path: Path):
        from ragdoc.document import Paragraph

        doc = Document(elements=[Paragraph(html_content="<p>Hello world</p>")])
        runner = CliRunner()
        output_dir = tmp_path / "out"
        with patch("ragdoc.cli.load", new_callable=AsyncMock) as mock_load:
            mock_load.return_value = doc
            result = runner.invoke(
                document_processing,
                ["chunk", str(html_file_path), "--output", str(output_dir), "--format", "html"],
            )

        assert result.exit_code == 0
        html_files = list(output_dir.glob("*.html"))
        assert len(html_files) == 1
