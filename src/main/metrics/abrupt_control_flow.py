from tree_sitter import Node

from src.main.metrics import metric


@metric("Abrupt Control Flow")
def abrupt_control_flow(root: Node) -> int:
    def _count(n: Node) -> int:
        hits = 0
        if n.type == "jump_expression" and n.child_count:
            child = n.children[0]
            kw = child.text.decode("utf8") if getattr(child, "text", None) else None
            if kw in ("break", "continue"):
                hits = 1
        return hits + sum(_count(c) for c in n.children)

    return _count(root)
