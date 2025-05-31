from typing import List, Optional
import pytest

from src.main.metrics.program_size import program_size


class MockNode:
    """
    Minimal mock for tree_sitter.Node to test program_size.
    Each node can have a .children and .type.
    """

    def __init__(
        self,
        type_: str,
        children: Optional[List["MockNode"]] = None,
        text: Optional[bytes] = None,  # not used, just for compatibility
    ):
        self.type = type_
        self.children = children if children is not None else []
        self.child_count = len(self.children)
        self.text = text


@pytest.mark.parametrize(
    "tree, expected",
    [
        pytest.param(
            # Root + 2 children
            MockNode("root", [MockNode("statement"), MockNode("property_declaration")]),
            3,  # root + 2 children
            id="root-two-children",
        ),
        pytest.param(
            # Root + 1 child
            MockNode("root", [MockNode("if_expression")]),
            2,  # root + if_expression
            id="root-single-child",
        ),
        pytest.param(
            # Root + 2 children, one with a child
            MockNode(
                "root",
                [
                    MockNode("if_expression"),
                    MockNode("when_expression", [MockNode("statement")]),
                ],
            ),
            4,  # root + if_expression + when_expression + statement
            id="nested-one-level",
        ),
        pytest.param(
            # Nested structure: while_statement > if_expression
            MockNode(
                "root", [MockNode("while_statement", [MockNode("if_expression")])]
            ),
            3,  # root + while_statement + if_expression
            id="nested-while-if",
        ),
        pytest.param(
            # Deep nesting: if > for > while
            MockNode(
                "if_expression",
                [MockNode("for_statement", [MockNode("while_statement")])],
            ),
            3,  # if_expression + for_statement + while_statement
            id="deep-nested",
        ),
        pytest.param(
            # Deep chain: root > a > b > c
            MockNode("root", [MockNode("a", [MockNode("b", [MockNode("c")])])]),
            4,  # root, a, b, c
            id="long-chain",
        ),
        pytest.param(
            # Only root node, no children
            MockNode("root"),
            1,
            id="single-node",
        ),
        pytest.param(
            # Root with several children
            MockNode(
                "root", [MockNode("catch_clause"), MockNode("do_while_statement")]
            ),
            3,  # root + 2 children
            id="root-multi-children",
        ),
        pytest.param(
            # Multi-nested: for > if > while > when
            MockNode(
                "for_statement",
                [
                    MockNode(
                        "if_expression",
                        [MockNode("while_statement", [MockNode("when_expression")])],
                    )
                ],
            ),
            4,  # for_statement + if_expression + while_statement + when_expression
            id="multi-nested",
        ),
        pytest.param(
            # Wide tree: root with 3 leaves
            MockNode("root", [MockNode("leaf1"), MockNode("leaf2"), MockNode("leaf3")]),
            4,  # root + 3 leaves
            id="wide-tree",
        ),
        pytest.param(
            # Empty root (no children)
            MockNode("root", []),
            1,
            id="empty-root",
        ),
    ],
)
def test_program_size(tree: MockNode, expected: int) -> None:
    """
    Tests program_size metric on various synthetic AST trees.
    """
    assert program_size(tree) == expected
