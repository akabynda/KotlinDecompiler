import math
from collections import Counter

from entropy.common import _merge, _tokens


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
