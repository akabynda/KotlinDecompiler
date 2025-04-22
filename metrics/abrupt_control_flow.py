from tree_sitter import Node

from metrics import metric


@metric("Abrupt Control Flow")
def abrupt_control_flow(root: Node) -> int:
    def _count(node: Node) -> int:
        hits = 1 if node.type == "jump_expression" else 0
        return hits + sum(_count(c) for c in node.children)

    return _count(root)
