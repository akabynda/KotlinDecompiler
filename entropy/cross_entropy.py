import math

from entropy.common import _merge, dist


def cross_entropy(p_src: str, q_src: str, eps=1e-12):
    p, q = dist(p_src), dist(q_src)
    return sum(q.get(k, eps) * math.log2(1 / max(p.get(k, eps), eps))
               for k in _merge(p, q))
