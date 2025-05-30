from tree_sitter import Node

from src.main.metrics import metric


@metric("Conditional Complexity")
def conditional_complexity(root: Node) -> float:
    def _complex(node: Node) -> float:
        if not node.children:
            return 1
        if node.type == "unary_expression":
            return 0.5 + _complex(node.children[-1])
        if node.type == "binary_expression" and len(node.children) >= 3:
            op = node.children[1].text.decode("utf8")
            left, right = node.children[0], node.children[2]
            if op in ("<", ">", "<=", ">=", "==", "!="):
                return 0.5 + _complex(left) + _complex(right)
            if op in ("&&", "||"):
                return 1 + _complex(left) + _complex(right)
        return sum(_complex(ch) for ch in node.children)

    def _conditions(node: Node):
        conds = []
        if node.type == "if_expression":
            cond = node.child_by_field_name("condition")
            if cond:
                conds.append(cond)
        for ch in node.children:
            conds.extend(_conditions(ch))
        return conds

    cond_list = _conditions(root)
    return (sum(_complex(c) for c in cond_list) / len(cond_list)) if cond_list else 0.0
