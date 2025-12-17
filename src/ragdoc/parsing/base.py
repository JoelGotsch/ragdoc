from functools import singledispatch

from ragdoc.document import Document


@singledispatch
def load_file(file_obj) -> Document: ...
