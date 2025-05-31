import pytest

from src.main.utils.make_prompt import make_prompt, wrap_as_row, to_bytecode


# Minimal mock of the Row object, matching expected API
class Row:
    def __init__(self, kt_path, kt_source, bytecode):
        self.kt_path = kt_path
        self.kt_source = kt_source
        self.bytecode = bytecode


@pytest.mark.parametrize(
    "kt_path, kt_source, bytecode, expected_prompt, expected_target",
    [
        pytest.param(
            "src/test/Foo.kt",
            "fun main() {\n  println(42)\n}\n",  # With trailing newline
            "public void main() {}",
            (
                "<|im_start|>system\n"
                "Convert the following JVM byte-code into **Kotlin source code**.\n"
                "Output Kotlin code only. Do not add any explanations.\n"
                "<|im_end|>\n"
                "<|im_start|>user\n"
                "public void main() {}\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            ),
            "```kotlin\nfun main() {\n  println(42)\n}\n```\n<|im_end|>\n",
            id="basic",
        ),
        pytest.param(
            "A.kt",
            "class A {}",
            "class A",
            (
                "<|im_start|>system\n"
                "Convert the following JVM byte-code into **Kotlin source code**.\n"
                "Output Kotlin code only. Do not add any explanations.\n"
                "<|im_end|>\n"
                "<|im_start|>user\n"
                "class A\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            ),
            "```kotlin\nclass A {}\n```\n<|im_end|>\n",
            id="no-trailing-newline",
        ),
        pytest.param(
            "Path/Bar.kt",
            "val x = 123",
            "Compiled: x=123",
            (
                "<|im_start|>system\n"
                "Convert the following JVM byte-code into **Kotlin source code**.\n"
                "Output Kotlin code only. Do not add any explanations.\n"
                "<|im_end|>\n"
                "<|im_start|>user\n"
                "Compiled: x=123\n"
                "<|im_end|>\n"
                "<|im_start|>assistant\n"
            ),
            "```kotlin\nval x = 123\n```\n<|im_end|>\n",
            id="single-line-kt",
        ),
    ],
)
def test_make_prompt_format(
    kt_path, kt_source, bytecode, expected_prompt, expected_target
):
    """
    Test that make_prompt returns the prompt and target formatted exactly as required.
    """
    row = Row(kt_path, kt_source, bytecode)
    result = make_prompt(row)
    # Check joined prompt
    assert result["text"].startswith(expected_prompt)
    assert expected_target in result["text"]
    # Check fields
    assert result["kt_path"] == kt_path
    assert result["bytecode"] == bytecode
    assert result["kt_source"] == kt_source


def test_make_prompt_trailing_newline_handling():
    """
    The Kotlin source should not lose its final newline inside the code block.
    """
    row = Row("Test.kt", "val foo = 42\n", "some bytecode")
    result = make_prompt(row)
    # Should not have two trailing newlines inside the code block
    assert result["text"].count("```\n<|im_end|>") == 1
    # There is always exactly one newline before the code block ends
    assert result["text"].split("```kotlin\n")[1].endswith("\n```\n<|im_end|>\n")


def test_make_prompt_multiline_bytecode():
    """
    The user prompt should include the bytecode block exactly.
    """
    bytecode = "line1\nline2\nline3"
    row = Row("Foo.kt", "fun f() = 0", bytecode)
    result = make_prompt(row)
    # User prompt ends after all bytecode lines
    assert "<|im_start|>user\nline1\nline2\nline3\n<|im_end|>" in result["text"]


def test_make_prompt_with_wrap_and_to_bytecode():
    """
    Test integration: wrap_as_row and to_bytecode cooperate as expected.
    """
    example = {
        "kt_path": "Q.kt",
        "kt_source": "fun q() {}",
        "classes": [{"javap": "first"}, {"javap": "second"}],
    }
    row = wrap_as_row(example)
    # to_bytecode called by wrap_as_row
    assert row.bytecode == "first\nsecond"
    # Now make_prompt returns prompt with all expected components
    prompt_dict = make_prompt(row)
    assert "first\nsecond" in prompt_dict["text"]
    assert "fun q() {}" in prompt_dict["text"]


@pytest.mark.parametrize(
    "kt_source",
    [
        "fun foo() {}\n",
        "fun bar() {}",
        "",
    ],
)
def test_make_prompt_code_block_exact(kt_source):
    """
    The code block output should wrap exactly the Kotlin source.
    """
    row = Row("Any.kt", kt_source, "irrelevant")
    result = make_prompt(row)
    code_block = f"```kotlin\n{kt_source.rstrip()}\n```\n<|im_end|>\n"
    assert code_block in result["text"]


@pytest.mark.parametrize(
    "example, expected_bytecode",
    [
        (
            {
                "kt_path": "Demo.kt",
                "kt_source": "val x = 0",
                "classes": [{"javap": "abc"}, {"javap": "def"}],
            },
            "abc\ndef",
        ),
        (
            {
                "kt_path": "Empty.kt",
                "kt_source": "",
                "classes": [],
            },
            "",
        ),
        (
            {
                "kt_path": "One.kt",
                "kt_source": "a",
                "classes": [{"javap": "unique"}],
            },
            "unique",
        ),
    ],
)
def test_to_bytecode_and_wrap_as_row(example, expected_bytecode):
    """
    to_bytecode and wrap_as_row should assemble bytecode string correctly.
    """
    assert to_bytecode(example) == expected_bytecode
    row = wrap_as_row(example)
    assert row.bytecode == expected_bytecode
    assert row.kt_path == example["kt_path"]
    assert row.kt_source == example["kt_source"]
