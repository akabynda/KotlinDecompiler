from typing import List, Optional
import pytest

from src.main.metrics.labeled_blocks import labeled_blocks


class MockNode:
    """
    Minimal mock for tree_sitter.Node to test labeled_blocks.
    Each node can have a .children.
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
            # Single label node
            MockNode("label"),
            1,
            id="single-label",
        ),
        pytest.param(
            # Labeled expression node at root
            MockNode("labeled_expression"),
            1,
            id="single-labeled_expression",
        ),
        pytest.param(
            # Label_definition node
            MockNode("label_definition"),
            1,
            id="single-label_definition",
        ),
        pytest.param(
            # Root with one label child
            MockNode("root", [MockNode("label")]),
            1,
            id="one-label-child",
        ),
        pytest.param(
            # Nested labeled nodes
            MockNode(
                "root",
                [
                    MockNode(
                        "statement",
                        [
                            MockNode(
                                "labeled_expression", [MockNode("label_definition")]
                            )
                        ],
                    )
                ],
            ),
            2,  # labeled_expression + label_definition
            id="nested-labeled",
        ),
        pytest.param(
            # Multiple siblings: 2 labels, 1 labeled_expression, 1 irrelevant
            MockNode(
                "root",
                [
                    MockNode("label"),
                    MockNode("label"),
                    MockNode("labeled_expression"),
                    MockNode("statement"),
                ],
            ),
            3,
            id="multiple-labeled-siblings",
        ),
        pytest.param(
            # No labeled nodes
            MockNode(
                "root",
                [
                    MockNode("statement"),
                    MockNode("block"),
                    MockNode("identifier"),
                ],
            ),
            0,
            id="no-labeled",
        ),
        pytest.param(
            # Deeply nested with irrelevant nodes
            MockNode(
                "root",
                [
                    MockNode(
                        "statement",
                        [
                            MockNode(
                                "block",
                                [
                                    MockNode("label"),
                                    MockNode("expression"),
                                    MockNode("labeled_expression"),
                                ],
                            )
                        ],
                    )
                ],
            ),
            2,
            id="deep-mixed",
        ),
        pytest.param(
            # Empty root
            MockNode("root", []),
            0,
            id="empty-root",
        ),
    ],
)
def test_labeled_blocks(tree: MockNode, expected: int) -> None:
    """
    Tests labeled_blocks metric on various synthetic AST trees.
    """
    assert labeled_blocks(tree) == expected
