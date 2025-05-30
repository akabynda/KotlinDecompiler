from typing import List, Optional
import pytest

from main.metrics.abrupt_control_flow import abrupt_control_flow

class MockNode:
    """
    Minimal mock for tree_sitter.Node to test abrupt_control_flow.
    Each token node may have a .text (bytes) field.
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
            # One break and one continue at the root
            MockNode("root", [
                MockNode("jump_expression", [
                    MockNode("token", text=b"break")
                ]),
                MockNode("jump_expression", [
                    MockNode("token", text=b"continue")
                ]),
            ]),
            2,
            id="break-and-continue"
        ),
        pytest.param(
            # Nested: only one break inside statement
            MockNode("root", [
                MockNode("statement", [
                    MockNode("jump_expression", [
                        MockNode("token", text=b"break")
                    ])
                ])
            ]),
            1,
            id="nested-break"
        ),
        pytest.param(
            # No jump_expression nodes
            MockNode("root", [
                MockNode("statement")
            ]),
            0,
            id="no-jump"
        ),
        pytest.param(
            # jump_expression but with return (not break/continue)
            MockNode("jump_expression", [
                MockNode("token", text=b"return")
            ]),
            0,
            id="return-only"
        ),
        pytest.param(
            # Multiple levels, several breaks/continues
            MockNode("root", [
                MockNode("jump_expression", [
                    MockNode("token", text=b"break")
                ]),
                MockNode("jump_expression", [
                    MockNode("token", text=b"continue")
                ]),
                MockNode("statement", [
                    MockNode("jump_expression", [
                        MockNode("token", text=b"continue")
                    ])
                ]),
            ]),
            3,
            id="multiple-breaks-continues"
        ),
        pytest.param(
            # jump_expression node with no children
            MockNode("jump_expression", []),
            0,
            id="empty-jump_expression"
        ),
        pytest.param(
            # jump_expression node with first child missing text
            MockNode("jump_expression", [MockNode("token")]),
            0,
            id="jump_expression-no-text"
        ),
    ]
)
def test_abrupt_control_flow(tree: MockNode, expected: int) -> None:
    """
    Tests abrupt_control_flow metric on various synthetic AST trees.
    """
    assert abrupt_control_flow(tree) == expected
