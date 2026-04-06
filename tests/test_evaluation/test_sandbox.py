"""Tests for the code execution sandbox utilities."""

from averyml.evaluation.benchmarks.livecodebench_utils import (
    clean_if_name,
    make_function,
    post_process_code,
    reliability_guard,
)
from averyml.evaluation.sandbox import CodeSandbox


class TestCodeSandbox:
    def test_construction(self):
        sandbox = CodeSandbox(timeout=10.0)
        assert sandbox.timeout == 10.0

    def test_default_timeout(self):
        sandbox = CodeSandbox()
        assert sandbox.timeout == 6.0


class TestCleanIfName:
    def test_unwraps_main_block(self):
        code = '''
def foo():
    pass

if __name__ == "__main__":
    foo()
'''
        result = clean_if_name(code.strip())
        assert "__name__" not in result
        assert "foo()" in result

    def test_leaves_regular_code(self):
        code = "x = 1\nprint(x)"
        result = clean_if_name(code)
        assert result == code


class TestMakeFunction:
    def test_wraps_code(self):
        code = "x = 1\nprint(x)"
        result = make_function(code)
        assert "wrapped_function" in result

    def test_preserves_imports(self):
        code = "import math\nprint(math.pi)"
        result = make_function(code)
        assert "wrapped_function" in result


class TestPostProcessCode:
    def test_strips_markdown_fences(self):
        code = "```python\nprint('hello')\n```"
        result = post_process_code(code)
        assert "```" not in result
        assert "print('hello')" in result

    def test_handles_html_code_tags(self):
        code = "<code>print('hello')</code>"
        result = post_process_code(code)
        assert "<code>" not in result
        assert "</code>" not in result
