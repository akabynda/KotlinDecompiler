from typing import List, Optional
import pytest

from src.main.metrics.chapin_q import chapin_q

class MockNode:
    """
    Minimal mock for tree_sitter.Node to test chapin_q.
    Each node can have a .text (bytes) and .children.
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
            # One var declared, used once (M=1, used=1, declared=1, C=0, T=0, P=0)
            MockNode("root", [
                MockNode("var_modifier", [
                    MockNode("identifier", text=b"x")
                ], text=None),
                MockNode("identifier", text=b"x")
            ]),
            2.0,  # P + 2*M + 3*C + 0.5*T == 0 + 2*1 + 0 + 0
            id="simple-declare-use"
        ),
        pytest.param(
            # Assignment to y, then using x, both not declared (P=1, M=1, C=0, T=0)
            MockNode("root", [
                MockNode("assignment_expression", [
                    MockNode("identifier", text=b"y")
                ], text=None),
                MockNode("identifier", text=b"x"),
            ]),
            3.0,  # P=1 (x used not declared), M=1, 2*1 + 1 = 3
            id="assign-and-use"
        ),
        pytest.param(
            # Using only undeclared variables, two times (P=2, M=0, C=0, T=0)
            MockNode("root", [
                MockNode("identifier", text=b"a"),
                MockNode("identifier", text=b"b"),
            ]),
            2.0,
            id="undeclared-use"
        ),
        pytest.param(
            # If statement, declared variable used (C=1)
            MockNode("if_expression", [
                MockNode("var_modifier", [
                    MockNode("identifier", text=b"z")
                ]),
                MockNode("identifier", text=b"z")
            ]),
            5.0,  # P=0, M=1, C=1, T=0 → 0 + 2*1 + 3*1 = 5
            id="if-declare-use"
        ),
        pytest.param(
            # Used and declared variables, control flow, extra used variable (T>0)
            MockNode("root", [
                MockNode("var_modifier", [
                    MockNode("identifier", text=b"v")
                ]),
                MockNode("identifier", text=b"v"),
                MockNode("identifier", text=b"w"),
                MockNode("if_expression", [])
            ]),
            6.0,  # declared: v, used: v, w => P=1 (w not declared), M=1, C=1, T=0 (len(used)-P-M=0)
            id="declare-use-extra"
        ),
        pytest.param(
            # More used vars than declared and P, non-negative T
            MockNode("root", [
                MockNode("identifier", text=b"a"),
                MockNode("identifier", text=b"b"),
                MockNode("var_modifier", [
                    MockNode("identifier", text=b"v")
                ]),
            ]),
            4.0,  # P=2 (a,b), M=1, C=0, T=0 (max(3-2-1, 0)); 2+2*1+0+0=4.0
            id="undeclared-and-declared"
        ),
    ]
)
def test_chapin_q(tree: MockNode, expected: float) -> None:
    """
    Tests chapin_q metric on various synthetic AST trees.
    """
    assert chapin_q(tree) == expected
