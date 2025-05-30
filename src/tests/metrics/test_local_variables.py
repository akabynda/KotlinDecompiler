from typing import List, Optional
import pytest

from src.main.metrics.local_variables import local_variables

class MockNode:
    """
    Minimal mock for tree_sitter.Node to test local_variables.
    Each node can have a .children and .type.
    """
    def __init__(
        self,
        type_: str,
        children: Optional[List['MockNode']] = None,
        text: Optional[bytes] = None,  # not used, but keep for style
    ):
        self.type = type_
        self.children = children if children is not None else []
        self.child_count = len(self.children)
        self.text = text

@pytest.mark.parametrize(
    "tree, expected",
    [
        pytest.param(
            # No function, no variables
            MockNode("root", [
                MockNode("property_declaration"),
                MockNode("statement"),
            ]),
            0,
            id="no-fn-no-var"
        ),
        pytest.param(
            # One local variable inside a function
            MockNode("function_declaration", [
                MockNode("property_declaration"),
            ]),
            1,
            id="one-var-in-fn"
        ),
        pytest.param(
            # One property_declaration outside, one inside function
            MockNode("root", [
                MockNode("property_declaration"),
                MockNode("function_definition", [
                    MockNode("property_declaration"),
                ])
            ]),
            1,
            id="one-var-in-fn-outside-does-not-count"
        ),
        pytest.param(
            # Nested function: variable in each function
            MockNode("function_definition", [
                MockNode("property_declaration"),
                MockNode("function_declaration", [
                    MockNode("property_declaration")
                ])
            ]),
            2,
            id="nested-fn-each-var-counts"
        ),
        pytest.param(
            # Multiple local variables in function
            MockNode("function_declaration", [
                MockNode("property_declaration"),
                MockNode("property_declaration"),
                MockNode("statement"),
            ]),
            2,
            id="two-vars-in-fn"
        ),
        pytest.param(
            # Deeply nested property_declaration in fn
            MockNode("function_definition", [
                MockNode("block", [
                    MockNode("statement", [
                        MockNode("property_declaration"),
                        MockNode("statement")
                    ])
                ])
            ]),
            1,
            id="deep-var-in-fn"
        ),
        pytest.param(
            # No property_declaration at all
            MockNode("function_declaration", [
                MockNode("statement"),
                MockNode("block"),
            ]),
            0,
            id="no-var-in-fn"
        ),
        pytest.param(
            # Variable in nested function only
            MockNode("function_declaration", [
                MockNode("statement"),
                MockNode("function_definition", [
                    MockNode("property_declaration")
                ])
            ]),
            1,
            id="nested-var-only"
        ),
        pytest.param(
            # Empty tree
            MockNode("root", []),
            0,
            id="empty-tree"
        ),
    ]
)
def test_local_variables(tree: MockNode, expected: int) -> None:
    """
    Tests local_variables metric on various synthetic AST trees.
    """
    assert local_variables(tree) == expected
