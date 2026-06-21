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

    In this example we will use the output of [Mineru](https://github.com/opendatalab/mineru)'s hybrid output (_middle.json file) and show how to parse, process, split, summarize, and render it using RagDoc.
    """)
    return


@app.cell
def _():
    from ragdoc.parsing.mineru import MinerUParser

    return


if __name__ == "__main__":
    app.run()
