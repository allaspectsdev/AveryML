"""Tests for the minimal synthesis filters."""

from averyml.synthesis.filters import (
    apply_minimal_filters,
    extract_code_block,
    is_empty_response,
    is_single_line_stub,
)


class TestExtractCodeBlock:
    def test_python_block(self):
        text = '```python\nprint("hello")\n```'
        assert extract_code_block(text) == 'print("hello")\n'

    def test_no_block(self):
        text = 'print("hello")'
        assert extract_code_block(text) == text

    def test_multiple_blocks(self):
        text = '```python\nfirst\n```\nsome text\n```python\nsecond\n```'
        assert extract_code_block(text) == "second\n"


class TestIsEmptyResponse:
    def test_empty_string(self):
        assert is_empty_response("") is True

    def test_whitespace(self):
        assert is_empty_response("   \n  ") is True

    def test_content(self):
        assert is_empty_response("def solve(): pass") is False

    def test_none(self):
        assert is_empty_response(None) is True


class TestIsSingleLineStub:
    def test_pass(self):
        assert is_single_line_stub("```python\npass\n```") is True

    def test_ellipsis(self):
        assert is_single_line_stub("```python\n...\n```") is True

    def test_return_none(self):
        assert is_single_line_stub("```python\nreturn None\n```") is True

    def test_real_code(self):
        assert is_single_line_stub("```python\nx = 1\nprint(x)\n```") is False

    def test_single_meaningful_line(self):
        assert is_single_line_stub("```python\nprint(solve(arr))\n```") is False


class TestApplyMinimalFilters:
    def test_removes_empty(self):
        samples = [
            {"response": ""},
            {"response": "def solve(): return 42"},
        ]
        result = apply_minimal_filters(samples)
        assert len(result) == 1
        assert result[0]["response"] == "def solve(): return 42"

    def test_removes_stubs(self):
        samples = [
            {"response": "```python\npass\n```"},
            {"response": "```python\ndef solve(n):\n    return n * 2\n```"},
        ]
        result = apply_minimal_filters(samples)
        assert len(result) == 1

    def test_keeps_valid(self):
        samples = [
            {"response": "```python\ndef solve(arr):\n    arr.sort()\n    return arr\n```"},
            {"response": "```python\nimport sys\nfor line in sys.stdin:\n    print(line)\n```"},
        ]
        result = apply_minimal_filters(samples)
        assert len(result) == 2
