from tree_sitter import Node

from src.main.metrics import metric


@metric("Local Variables")
def local_variables(root: Node) -> int:
    def _count(node: Node, in_fn: bool = False) -> int:
        if node.type in ("function_declaration", "function_definition"):
            in_fn = True
        hits = 1 if in_fn and node.type == "property_declaration" else 0
        return hits + sum(_count(c, in_fn) for c in node.children)

    return _count(root)
