from tree_sitter import Node

from src.main.metrics import metric

_DECISIONS = {
    "if_expression",
    "when_expression",
    "while_statement",
    "do_while_statement",
    "for_statement",
    "catch_clause",
    "binary_expression"
}


@metric("Cyclomatic Complexity")
def cyclomatic_complexity(root: Node):
    def _cnt(n: Node):
        d = 1 if n.type in _DECISIONS else 0
        return d + sum(_cnt(c) for c in n.children)

    return _cnt(root) + 1
