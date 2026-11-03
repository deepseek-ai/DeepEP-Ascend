"""Test the real byte parser without importing the NPU package or extension."""

import ast
import re
from pathlib import Path

import pytest


def _load_parse_num_bytes():
    source = Path(__file__).resolve().parents[2] / 'deep_ep/utils/testing.py'
    module = ast.parse(source.read_text(encoding='utf-8'), filename=str(source))
    parser = next(node for node in module.body
                  if isinstance(node, ast.FunctionDef) and node.name == 'parse_num_bytes')
    namespace = {'re': re}
    exec(compile(ast.Module(body=[parser], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['parse_num_bytes']


parse_num_bytes = _load_parse_num_bytes()


@pytest.mark.parametrize(
    ('text', 'expected'),
    (
        ('1K', 1 << 10),
        ('1.5 MiB', 3 << 19),
        ('9007199254740993', 9007199254740993),
        ('1.0009765624999999999K', 1024),
    ),
)
def test_parse_num_bytes_exact(text, expected):
    assert parse_num_bytes(text) == expected


@pytest.mark.parametrize('text', ('0', '0.0001', '-1G', '1.2.3M', '1P'))
def test_parse_num_bytes_rejects_invalid_values(text):
    with pytest.raises(ValueError):
        parse_num_bytes(text)
