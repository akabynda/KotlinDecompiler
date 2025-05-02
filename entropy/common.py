from __future__ import annotations

from collections import Counter

from tree_sitter_languages import get_parser

_parser = get_parser("kotlin")


def _leaves(node):
    stack = [node]
    while stack:
        n = stack.pop()
        if n.child_count == 0:
            yield n
        else:
            stack.extend(reversed(n.children))


def _tokens(src: str) -> list[str]:
    tree = _parser.parse(src.encode("utf8"))
    return [n.text.decode("utf8") for n in _leaves(tree.root_node)]


def dist(src: str) -> dict[str, float]:
    toks = _tokens(src)
    tot = len(toks)
    c = Counter(toks)
    return {t: c[t] / tot for t in c}


def _merge(p, q):
    s = set(p)
    s.update(q)
    return s
