import pytest

from main.utils.extract_kotlin import extract_kotlin


@pytest.mark.parametrize(
    "text, expected",
    [
        pytest.param(
            # Kotlin code fenced block (with "kotlin" lang)
            "```kotlin\nfun main() { println(42) }\n```",
            "fun main() { println(42) }",
            id="fenced-kotlin"
        ),
        pytest.param(
            # Kotlin fenced block with language and garbage after 'kotlin'
            "```kotlin extra\nfun main() {}\n```",
            "fun main() {}",
            id="fenced-kotlin-extra"
        ),
        pytest.param(
            # Code fenced block, no language specified
            "```\nval x = 1\nprintln(x)\n```",
            "val x = 1\nprintln(x)",
            id="fenced-generic"
        ),
        pytest.param(
            # Fenced block with other language - should not match first pattern, but matches second
            "```python\nprint(1)\n```\n```kotlin\nval y = 2\n```",
            "val y = 2",
            id="fenced-otherlang-first"
        ),
        pytest.param(
            # Markdown header with '### Kotlin' and code after
            "### Kotlin\nval foo = 10\nbar()\n###",
            "val foo = 10\nbar()",
            id="header-style"
        ),
        pytest.param(
            # Markdown header with '### Kotlin', no closing header
            "### Kotlin\nfun x() = 123",
            "fun x() = 123",
            id="header-no-end"
        ),
        pytest.param(
            # Fenced block, missing closing ```
            "```kotlin\nprintln(\"hello\")",
            "println(\"hello\")",
            id="fenced-no-end"
        ),
        pytest.param(
            # Block with only whitespace inside
            "```kotlin\n   \n```",
            "",
            id="only-whitespace"
        ),
        pytest.param(
            # Multiple code blocks, match first only
            "```kotlin\nfirst()\n```\n```kotlin\nsecond()\n```",
            "first()",
            id="multiple-kotlin-blocks"
        ),
        pytest.param(
            # No code block, just plain text
            "This is just text.",
            "",
            id="no-code"
        ),
        pytest.param(
            # Non-Kotlin fenced, but header with Kotlin
            "```python\nnot kotlin\n```\n### Kotlin\nreal_kotlin()\n###",
            "not kotlin",
            id="prefer-code"
        ),
        pytest.param(
            # No code blocks, header with no trailing ###
            "### Kotlin\nsome code without end",
            "some code without end",
            id="header-no-trailing"
        ),
        pytest.param(
            # Newline after header, then code, then text but no new header
            "### Kotlin\nfun foo() {}\nbar()\nmore text",
            "fun foo() {}\nbar()\nmore text",
            id="header-followed-by-more"
        ),
        pytest.param(
            # Fenced block with CRLF newlines
            "```kotlin\r\nfun foo() {}\r\n```",
            "fun foo() {}",
            id="fenced-kotlin-crlf"
        ),
    ]
)
def test_extract_kotlin(text, expected):
    """
    Tests extract_kotlin function on various formatted texts.
    """
    assert extract_kotlin(text) == expected
