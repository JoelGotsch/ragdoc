NOTEBOOKS_PY := $(wildcard docs/notebooks/*.py)
NOTEBOOKS_WASM := $(patsubst docs/notebooks/%.py,docs/notebooks/%/index.html,$(NOTEBOOKS_PY))

.PHONY: help notebooks html serve clean

help:
	@echo "make notebooks   Export all marimo notebooks to WASM HTML"
	@echo "make html        Build the documentation site (runs notebooks first)"
	@echo "make serve       Serve the MkDocs site with live reload"
	@echo "make clean       Remove generated WASM dirs and the site/ directory"

notebooks: $(NOTEBOOKS_WASM)

docs/notebooks/%/index.html: docs/notebooks/%.py
	uv run marimo export html-wasm $< -o docs/notebooks/$* --mode run

html: notebooks
	uv run mkdocs build
	touch site/.nojekyll

serve: notebooks
	uv run mkdocs serve

clean:
	@for d in $(dir $(NOTEBOOKS_WASM)); do rm -rf "$$d"; done
	rm -rf site/
