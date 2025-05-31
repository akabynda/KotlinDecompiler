from tree_sitter import Node

from src.main.metrics import metric


@metric("Labeled Blocks")
def labeled_blocks(root: Node) -> int:
    def _count(node: Node) -> int:
        hits = (
            1 if node.type in ("label", "labeled_expression", "label_definition") else 0
        )
        return hits + sum(_count(c) for c in node.children)

    return _count(root)
