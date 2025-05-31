from typing import List, Optional
import pytest
import math

from src.main.metrics.halstead_common import counts, derived


class MockNode:
    """
    Minimal mock for tree_sitter.Node to test Halstead metrics.
    Each node can have a .text (bytes) and .children.
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
    "tree, expected_counts",
    [
        pytest.param(
            # One operator and one operand
            MockNode(
                "root",
                [
                    MockNode("identifier", text=b"x"),
                    MockNode("plus", text=b"+"),
                ],
            ),
            (1, 1, 1, 1),  # N1=1 (operator '+'), N2=1 (operand 'x'), n1=1, n2=1
            id="one-op-one-operand",
        ),
        pytest.param(
            # Multiple unique operators and operands
            MockNode(
                "root",
                [
                    MockNode("identifier", text=b"a"),
                    MockNode("plus", text=b"+"),
                    MockNode("integer_literal", text=b"42"),
                    MockNode("minus", text=b"-"),
                    MockNode("identifier", text=b"b"),
                ],
            ),
            (2, 3, 2, 3),  # N1=2 ('+', '-'), N2=3 ('a', '42', 'b'), n1=2, n2=2
            id="multiple-unique",
        ),
        pytest.param(
            # Repeated operators and operands
            MockNode(
                "root",
                [
                    MockNode("identifier", text=b"x"),
                    MockNode("plus", text=b"+"),
                    MockNode("identifier", text=b"x"),
                    MockNode("plus", text=b"+"),
                ],
            ),
            (2, 2, 1, 1),  # N1=2 ('+'), N2=2 ('x'), n1=1, n2=1
            id="repeated",
        ),
        pytest.param(
            # Non-operator/non-operand nodes
            MockNode(
                "root",
                [
                    MockNode("block", text=b"{"),
                    MockNode("if_expression", text=b"if"),
                ],
            ),
            (0, 0, 0, 0),  # Nothing recognized as operator or operand
            id="irrelevant-nodes",
        ),
        pytest.param(
            # No leaves
            MockNode("root", []),
            (0, 0, 0, 0),
            id="empty-tree",
        ),
    ],
)
def test_counts(tree: MockNode, expected_counts) -> None:
    """
    Tests Halstead counts for various ASTs.
    """
    assert counts(tree) == expected_counts


@pytest.mark.parametrize(
    "tree, expected",
    [
        pytest.param(
            # 1 operator ('+') and 2 unique operands ('a', 'b')
            MockNode(
                "root",
                [
                    MockNode("identifier", text=b"a"),
                    MockNode("plus", text=b"+"),
                    MockNode("identifier", text=b"b"),
                ],
            ),
            {
                "Halstead Length": 3,  # N = N1+N2 = 1+2
                "Halstead Vocabulary": 3,  # n = n1+n2 = 1+2
            },
            id="basic-volume",
        ),
        pytest.param(
            # Only one unique operator and operand
            MockNode(
                "root",
                [
                    MockNode("identifier", text=b"c"),
                    MockNode("plus", text=b"+"),
                ],
            ),
            {
                "Halstead Length": 2,  # 1+1
                "Halstead Vocabulary": 2,  # 1+1
                "Halstead Volume": 2 * math.log2(2),  # 2*1 = 2
            },
            id="minimal-volume",
        ),
        pytest.param(
            # No operators or operands
            MockNode("root", []),
            {
                "Halstead Length": 0,
                "Halstead Vocabulary": 0,
                "Halstead Volume": 0,
                "Halstead Difficulty": 0,
                "Halstead Effort": 0,
                "Halstead Time": 0,
                "Halstead Bugs": 0,
            },
            id="zero-case",
        ),
        pytest.param(
            # Example with several unique ops/operands
            MockNode(
                "root",
                [
                    MockNode("identifier", text=b"a"),
                    MockNode("plus", text=b"+"),
                    MockNode("minus", text=b"-"),
                    MockNode("identifier", text=b"b"),
                    MockNode("integer_literal", text=b"123"),
                    MockNode("plus", text=b"+"),
                ],
            ),
            None,  # we'll check that values make sense, not exact numbers
            id="complex-tree",
        ),
    ],
)
def test_derived(tree: MockNode, expected) -> None:
    """
    Tests Halstead derived metrics on various synthetic AST trees.
    """
    result = derived(tree)
    # For the first three test cases, check explicit values (the rest just check presence and types)
    if expected is not None:
        for k, v in expected.items():
            assert pytest.approx(result[k]) == v
    else:
        keys = [
            "Halstead Length",
            "Halstead Vocabulary",
            "Halstead Volume",
            "Halstead Difficulty",
            "Halstead Effort",
            "Halstead Time",
            "Halstead Bugs",
        ]
        for k in keys:
            assert k in result
            assert result[k] >= 0
