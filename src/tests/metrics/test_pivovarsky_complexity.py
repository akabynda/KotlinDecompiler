from typing import List, Optional
import pytest

from src.main.metrics.pivovarsky_complexity import pivovarsky


class MockNode:
    """
    Minimal mock for tree_sitter.Node to test pivovarsky.
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
            # No decision nodes at all (just root with statements)
            MockNode("root", [MockNode("statement"), MockNode("property_declaration")]),
            1,  # 0 decisions, so 0 + 1 + 0 = 1
            id="no-decisions",
        ),
        pytest.param(
            # Single if_expression at root
            MockNode("root", [MockNode("if_expression")]),
            3,  # 1 decision, depth 1; 1 + 1 + 1 = 3
            id="single-if",
        ),
        pytest.param(
            # Two sibling decisions at root (if, when)
            MockNode("root", [MockNode("if_expression"), MockNode("when_expression")]),
            5,  # (2 decisions at depth 1): 2 + 1 + 2 = 5
            id="two-sibling-decisions",
        ),
        pytest.param(
            # Nested if inside while
            MockNode(
                "root", [MockNode("while_statement", [MockNode("if_expression")])]
            ),
            6,  # while at depth 1 (cnt=1, sum=1), if at depth 2 (cnt=1, sum=2) => 2+1+3=6
            id="nested-if-in-while",
        ),
        pytest.param(
            # Deep nest: if > for > while
            MockNode(
                "if_expression",
                [MockNode("for_statement", [MockNode("while_statement")])],
            ),
            10,  # if (1,1), for (1,2), while (1,3): cnt=3, sum=6; 3+1+6=10
            id="deep-nested-decisions",
        ),
        pytest.param(
            # Mixed: if, and inside it a when and for (each nested)
            MockNode(
                "root",
                [
                    MockNode(
                        "if_expression",
                        [MockNode("when_expression", [MockNode("for_statement")])],
                    )
                ],
            ),
            10,  # if at 1, when at 2, for at 3; cnt=3, sum=6; 3+1+6=10
            id="mixed-nested",
        ),
        pytest.param(
            # No children at all
            MockNode("root"),
            1,
            id="empty-root",
        ),
        pytest.param(
            # Sibling catch_clause and do_while_statement
            MockNode(
                "root", [MockNode("catch_clause"), MockNode("do_while_statement")]
            ),
            5,  # both at depth 1: cnt=2, sum=2; 2+1+2=5
            id="catch-and-do-while",
        ),
        pytest.param(
            # Deeper nest: for > if > while > when
            MockNode(
                "for_statement",
                [
                    MockNode(
                        "if_expression",
                        [MockNode("while_statement", [MockNode("when_expression")])],
                    )
                ],
            ),
            15,  # for@1, if@2, while@3, when@4: cnt=4, sum=10; 4+1+10=15
            id="multi-nested",
        ),
    ],
)
def test_pivovarsky(tree: MockNode, expected: int) -> None:
    """
    Tests pivovarsky metric on various synthetic AST trees.
    """
    assert pivovarsky(tree) == expected
