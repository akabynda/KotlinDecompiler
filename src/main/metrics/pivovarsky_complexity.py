from tree_sitter import Node

from src.main.metrics import metric

_DECISIONS = {
    "if_expression",
    "when_expression",
    "while_statement",
    "do_while_statement",
    "for_statement",
    "catch_clause",
}


def _walk(n: Node, depth: int, res):
    if n.type in _DECISIONS:
        res[0] += 1
        res[1] += depth
        depth += 1
    for c in n.children:
        _walk(c, depth, res)


@metric("Pivovarsky N(G)")
def pivovarsky(root: Node):
    res = [0, 0]
    _walk(root, 1, res)
    return res[0] + 1 + res[1]
