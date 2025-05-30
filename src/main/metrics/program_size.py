from tree_sitter import Node

from src.main.metrics import metric


@metric("Program Size")
def program_size(root: Node) -> int:
    def _count(node: Node) -> int:
        return 1 + sum(_count(c) for c in node.children)

    return _count(root)
