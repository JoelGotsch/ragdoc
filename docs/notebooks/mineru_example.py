import marimo

__generated_with = "0.23.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Example Pipeline using Mineru

    In this example we will use the output of [Mineru](https://github.com/opendatalab/mineru)'s hybrid output (_middle.json file) and construct the parser for it. (The full parse → process → split → render walkthrough is in the pipeline walkthrough notebook.)
    """)
    return


@app.cell
def _():
    from ragdoc.parsing.mineru import MinerUParser

    parser = MinerUParser()
    print(f"MinerUParser ready — parses MinerU _middle.json files, sets document.parser={'mineru'!r}")
    return (MinerUParser, parser)


if __name__ == "__main__":
    app.run()
