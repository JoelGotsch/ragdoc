"""Tests for ExtractionSettings — prompt resolution, env plumbing, and the new Phase-4 knobs."""

from __future__ import annotations

from pathlib import Path

import pytest

from ragdoc.extraction.structured import EXTRACTION_SYSTEM_PROMPT, ExtractionSettings


def test_validator_no_longer_injects_builtin_default():
    """None must stay None so each extractor can resolve its OWN default prompt."""
    settings = ExtractionSettings()
    assert settings.system_prompt is None


def test_explicit_prompt_kept():
    settings = ExtractionSettings(system_prompt="MY PROMPT")
    assert settings.system_prompt == "MY PROMPT"
    assert settings.system_prompt != EXTRACTION_SYSTEM_PROMPT


def test_prompt_file_still_resolved(tmp_path: Path):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("  FROM FILE  \n", encoding="utf-8")
    settings = ExtractionSettings(system_prompt_file=str(prompt_file))
    assert settings.system_prompt == "FROM FILE"


def test_explicit_prompt_wins_over_file(tmp_path: Path):
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("FROM FILE", encoding="utf-8")
    settings = ExtractionSettings(system_prompt="EXPLICIT", system_prompt_file=str(prompt_file))
    assert settings.system_prompt == "EXPLICIT"


def test_max_union_size_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EXTRACTION_MAX_UNION_SIZE", "3")
    assert ExtractionSettings().max_union_size == 3


def test_halving_and_gleaning_knobs_exposed():
    settings = ExtractionSettings()
    assert settings.halving_max_depth == 3
    assert settings.halving_min_chars == 1000
    assert settings.gleaning is False
    assert settings.on_pattern_violation == "drop"


def test_gleaning_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("EXTRACTION_GLEANING", "true")
    assert ExtractionSettings().gleaning is True


def test_kg_max_union_size_env_is_dead(monkeypatch: pytest.MonkeyPatch):
    """The old KG_MAX_UNION_SIZE env var is deleted — it must have no effect."""
    monkeypatch.setenv("KG_MAX_UNION_SIZE", "1")
    assert ExtractionSettings().max_union_size == 10
