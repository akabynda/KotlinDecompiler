from __future__ import annotations

import math
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


def cross_entropy(p_src: str, q_src: str, eps=1e-12):
    p, q = dist(p_src), dist(q_src)
    return sum(q.get(k, eps) * math.log2(1 / max(p.get(k, eps), eps))
               for k in _merge(p, q))


def kl_div(p_src: str, q_src: str, eps=1e-12):
    p, q = dist(p_src), dist(q_src)
    return sum(q.get(k, eps) * math.log2(max(q.get(k, eps), eps) /
                                         max(p.get(k, eps), eps))
               for k in _merge(p, q))


def entropy(src: str):
    return cross_entropy(src, src)


def perplexity(p_src: str, q_src: str):
    return 2 ** cross_entropy(p_src, q_src)


def nid(p_src: str, q_src: str):
    h_p, h_q = entropy(p_src), entropy(q_src)
    return (kl_div(p_src, q_src) + kl_div(q_src, p_src)) / (h_p + h_q)


def conditional_entropy(p_src: str, q_src: str, eps=1e-12):
    def bigram(src: str):
        t = _tokens(src)
        if len(t) < 2:
            return {}, {}
        bi = Counter(zip(t, t[1:]))
        uni = Counter(t[:-1])
        total = sum(bi.values())
        b_prob = {k: v / total for k, v in bi.items()}
        u_prob = {k: uni[k] / total for k in uni}
        return b_prob, u_prob

    bp, up = bigram(p_src)
    bq, _ = bigram(q_src)
    res = 0.0
    for (a, b) in _merge(bp, bq):
        q = bq.get((a, b), eps)
        p_cond = bp.get((a, b), eps) / up.get(a, eps)
        res += q * math.log2(1 / max(p_cond, eps))
    return res
