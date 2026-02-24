import ast

import pytest

from flake8_no_module_forward_call.plugin import ERROR_MESSAGE, NoModuleForwardCallChecker


def errors(source: str):
    tree = ast.parse(source)
    checker = NoModuleForwardCallChecker(tree)
    return [(line, col, msg) for line, col, msg, _ in checker.run()]


# ---------------------------------------------------------------------------
# Cases that MUST be flagged
# ---------------------------------------------------------------------------


def test_simple_forward_call():
    src = "model.forward(x)"
    result = errors(src)
    assert len(result) == 1
    assert result[0] == (1, 0, ERROR_MESSAGE)


def test_forward_on_arbitrary_attribute():
    src = "self.encoder.forward(hidden)"
    result = errors(src)
    assert len(result) == 1


def test_forward_inside_function():
    src = """
def train_step(model, x):
    return model.forward(x)
"""
    result = errors(src)
    assert len(result) == 1
    assert result[0][0] == 3  # line 3


def test_multiple_forward_calls():
    src = """
out1 = encoder.forward(x)
out2 = decoder.forward(out1)
"""
    result = errors(src)
    assert len(result) == 2


def test_forward_call_in_class():
    src = """
class Trainer:
    def step(self, x):
        return self.model.forward(x)
"""
    result = errors(src)
    assert len(result) == 1


def test_forward_in_nested_call():
    src = "loss = criterion(model.forward(x), y)"
    result = errors(src)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Cases that must NOT be flagged
# ---------------------------------------------------------------------------


def test_normal_call_operator():
    src = "model(x)"
    assert errors(src) == []


def test_other_attribute_method():
    src = "model.backward(x)"
    assert errors(src) == []


def test_forward_method_definition():
    # defining forward is fine; only calling it is an error
    src = """
class MyModel:
    def forward(self, x):
        return x
"""
    assert errors(src) == []


def test_forward_attribute_access_without_call():
    # accessing the attribute without calling it is not flagged
    src = "fn = model.forward"
    assert errors(src) == []


def test_empty_module():
    assert errors("") == []


def test_column_offset():
    src = """
if True:
    model.forward(x)
"""
    result = errors(src)
    assert len(result) == 1
    assert result[0][1] == 4  # 4-space indent
