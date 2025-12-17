"""Notebook execution tests — run every notebook via app.run() and assert key outputs."""

import pytest


@pytest.mark.anyio
async def test_splitting_notebook():
    from docs.notebooks.splitting import app

    _outputs, defs = app.run()
    splits = defs["splits"]
    assert all("split_sequence" in d.metadata for d in splits)
    assert [d.metadata["split_sequence"] for d in splits] == list(range(1, len(splits) + 1))
    assert all(d.metadata["split_total"] == len(splits) for d in splits)
