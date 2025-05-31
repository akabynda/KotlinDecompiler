from typing import List, Optional
import pytest

from src.main.metrics.conditional_statements import conditional_statements


class MockNode:
    """
    Minimal mock for tree_sitter.Node to test conditional_statements.
    """

    def __init__(
        self,
        type_: str,
        children: Optional[List["MockNode"]] = None,
        text: Optional[bytes] = None,
    ):
        self.type = type_
        self.children = children if children is not None else []
        self.child_count = len(self.children)
        self.text = text


@pytest.mark.parametrize(
    "tree, expected",
    [
        pytest.param(
            # No children, not an if_expression
            MockNode("root", []),
            0,
            id="no-children",
        ),
        pytest.param(
            # Single if_expression, no children
            MockNode("if_expression", []),
            1,
            id="single-if",
        ),
        pytest.param(
            # One if_expression with child which is not if_expression
            MockNode("if_expression", [MockNode("identifier", [])]),
            1,
            id="if-with-nonif-child",
        ),
        pytest.param(
            # Nested if_expressions
            MockNode(
                "root",
                [
                    MockNode(
                        "if_expression",
                        [MockNode("if_expression", [MockNode("identifier", [])])],
                    )
                ],
            ),
            2,
            id="nested-if",
        ),
        pytest.param(
            # Several siblings, mix of if_expression and others
            MockNode(
                "root",
                [
                    MockNode("statement", []),
                    MockNode("if_expression", []),
                    MockNode("statement", []),
                    MockNode("if_expression", [MockNode("if_expression", [])]),
                ],
            ),
            3,
            id="multiple-ifs",
        ),
        pytest.param(
            # Deeply nested, no if_expression
            MockNode(
                "root",
                [MockNode("block", [MockNode("block", [MockNode("identifier", [])])])],
            ),
            0,
            id="deep-nonif",
        ),
    ],
)
def test_conditional_statements(tree: MockNode, expected: int) -> None:
    """
    Tests conditional_statements metric on various synthetic AST trees.
    """
    assert conditional_statements(tree) == expected
