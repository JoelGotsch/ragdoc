from functools import singledispatch
from typing import Iterator

from ragdoc.document import Document


@singledispatch
def load_file(file_obj) -> Document: ...
