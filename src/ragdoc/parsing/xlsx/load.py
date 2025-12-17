import pandas as pd
from pydantic import BaseModel, Field

from ragdoc.document import Document, Heading, Table


class ExcelConfig(BaseModel):
    """
    Used for configuring the loading of excel files.

    All parameters are passed to the :py:func:`pandas.read_excel` function.
    E.g. if you want to skip the first 10 rows and only read the first 33 rows, you can use:
    ExcelConfig(default_params={"skiprows": 10, "nrows": 33})

    Args:
        default_params (dict, optional): Default parameters for the pandas.read_excel function. Defaults to {}.
        sheet_params (dict[str, dict], optional): Parameters for specific sheets. Defaults to {}.
    The default_params are overruled by the sheet_params if the sheet name is present in the sheet_params dict.

    If sheet_params are provided and no default_params, then only the sheets are loaded for which parameters are provided.
    """

    default_params: dict | None = Field(default=None)
    sheet_params: dict[str, dict] = Field(default_factory=dict)


def generate_document(excel_file: pd.ExcelFile, config: ExcelConfig | None = None) -> Document:
    document = Document()
    config = ExcelConfig() if config is None else config
    for i, sheet_name in enumerate(excel_file.sheet_names):
        sheet_name = str(sheet_name)
        if not config.default_params and config.sheet_params and sheet_name not in config.sheet_params.keys():
            continue
        params = config.default_params or {} | config.sheet_params.get(sheet_name, {})
        df = pd.read_excel(excel_file, sheet_name=sheet_name, **params)
        document_heading = Heading(html_content=f"<h2>{sheet_name}</h2>", page=i)
        document_table = Table(html_content=df.to_html(), page=i)
        document.elements.append(document_heading)
        document.elements.append(document_table)
    document.parser = "xlsx"
    return document
