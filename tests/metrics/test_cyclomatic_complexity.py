from typing import List, Optional

import pytest

from main.metrics.cyclomatic_complexity import cyclomatic_complexity


class MockNode:
    """
    Minimal mock for tree_sitter.Node to test cyclomatic_complexity.
    """

    def __init__(
            self,
            type_: str,
            children: Optional[List['MockNode']] = None,
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
            # No decision nodes
            MockNode("root", []),
            1,
            id="no-decision-nodes"
        ),
        pytest.param(
            # Single if_expression node
            MockNode("if_expression", []),
            2,  # 1 (if_expression) + 1
            id="single-if"
        ),
        pytest.param(
            # Nested decision nodes
            MockNode("if_expression", [
                MockNode("while_statement", [
                    MockNode("for_statement", [])
                ])
            ]),
            4,  # 1 (if) + 1 (while) + 1 (for) + 1
            id="nested-decision-nodes"
        ),
        pytest.param(
            # Several siblings with some decision nodes
            MockNode("root", [
                MockNode("if_expression", []),
                MockNode("statement", []),
                MockNode("when_expression", []),
                MockNode("identifier", []),
            ]),
            3,  # if + when + 1
            id="multiple-sibling-decisions"
        ),
        pytest.param(
            # Mix of decision and non-decision, including binary_expression
            MockNode("root", [
                MockNode("binary_expression", [
                    MockNode("if_expression", []),
                    MockNode("statement", []),
                ]),
                MockNode("do_while_statement", []),
            ]),
            4,  # binary + if + do_while + 1
            id="decision-mix"
        ),
        pytest.param(
            # Deeply nested, no decision nodes
            MockNode("root", [
                MockNode("block", [
                    MockNode("block", [
                        MockNode("identifier", [])
                    ])
                ])
            ]),
            1,
            id="deep-nondecision"
        ),
        pytest.param(
            # Single catch_clause
            MockNode("catch_clause", []),
            2,
            id="single-catch"
        ),
    ]
)
def test_cyclomatic_complexity(tree: MockNode, expected: int) -> None:
    """
    Tests cyclomatic_complexity metric on various synthetic AST trees.
    """
    assert cyclomatic_complexity(tree) == expected
