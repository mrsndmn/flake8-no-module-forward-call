from __future__ import annotations

import ast
from collections.abc import Generator
from typing import Tuple

ERROR_CODE = "NMF001"
ERROR_MESSAGE = (
    "NMF001 Do not call .forward() directly; use model(inputs) instead so DDP backward hooks run"
)


class NoModuleForwardCallChecker:
    name = "flake8-no-module-forward-call"
    version = "0.1.0"

    def __init__(self, tree: ast.AST) -> None:
        self.tree = tree

    def run(self) -> Generator[Tuple[int, int, str, type], None, None]:
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute) and func.attr == "forward":
                    yield (node.lineno, node.col_offset, ERROR_MESSAGE, type(self))
