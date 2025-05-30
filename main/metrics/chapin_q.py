from tree_sitter import Node

from main.metrics import metric

_READ = {"identifier"}
_WRITE = {"assignment_expression", "var_modifier", "property_declaration"}
_CTRL = {"if_expression", "when_expression", "while_statement", "do_while_statement", "for_statement"}


@metric("Chapin Q")
def chapin_q(root: Node):
    P = M = C = T = 0
    declared = set()
    used = {}

    def _scan(n: Node):
        nonlocal P, M, C
        if n.type in _WRITE:
            name = n.children[0].text if n.children else None
            if name:
                declared.add(name)
                M += 1
        if n.type in _READ:
            used.setdefault(n.text, 0)
            used[n.text] += 1
        if n.type in _CTRL:
            C += 1
        for c in n.children:
            _scan(c)

    _scan(root)
    P = len([k for k in used if k not in declared])
    T = max(len(used) - P - M, 0)
    return P + 2 * M + 3 * C + 0.5 * T
