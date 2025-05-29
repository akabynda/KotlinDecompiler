import math
from collections import Counter

from tree_sitter_languages import get_parser


class Entropy:

    def __init__(self, language):
        self.parser = get_parser(language)

    def leaves(self, node):
        stack = [node]
        while stack:
            n = stack.pop()
            if n.child_count == 0:
                yield n
            else:
                stack.extend(reversed(n.children))

    def tokens(self, src: str) -> list[str]:
        tree = self.parser.parse(src.encode("utf8"))
        return [n.text.decode("utf8") for n in self.leaves(tree.root_node)]

    def dist(self, src: str) -> dict[str, float]:
        toks = self.tokens(src)
        tot = len(toks)
        c = Counter(toks)
        return {t: c[t] / tot for t in c}

    def merge(self, p, q):
        s = set(p)
        s.update(q)
        return s

    def conditional_entropy(self, p_src: str, q_src: str, eps=1e-12):
        def bigram(src: str):
            t = self.tokens(src)
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
        for (a, b) in self.merge(bp, bq):
            q = bq.get((a, b), eps)
            p_cond = bp.get((a, b), eps) / up.get(a, eps)
            res += q * math.log2(1 / max(p_cond, eps))
        return res

    def cross_entropy(self, p_src: str, q_src: str, eps=1e-12):
        p, q = self.dist(p_src), self.dist(q_src)
        return sum(q.get(k, eps) * math.log2(1 / max(p.get(k, eps), eps))
                   for k in self.merge(p, q))

    def kl_div(self, p_src: str, q_src: str, eps=1e-12):
        p, q = self.dist(p_src), self.dist(q_src)
        return sum(q.get(k, eps) * math.log2(max(q.get(k, eps), eps) /
                                             max(p.get(k, eps), eps))
                   for k in self.merge(p, q))

    def entropy(self, src: str):
        return self.cross_entropy(src, src)

    def jensen_shannon_divergence(self, p: dict[str, float], q: dict[str, float], eps=1e-12) -> float:
        m = {k: 0.5 * (p.get(k, 0.0) + q.get(k, 0.0)) for k in self.merge(p, q)}
        kl_pm = 0.0
        kl_qm = 0.0
        for k in self.merge(p, q):
            p_val = p.get(k, eps)
            q_val = q.get(k, eps)
            m_val = m.get(k, eps)
            kl_pm += p_val * math.log2(p_val / m_val)
            kl_qm += q_val * math.log2(q_val / m_val)
        return 0.5 * (kl_pm + kl_qm)

    def jensen_shannon_distance(self, p_src: str, q_src: str) -> float:
        p_dist = self.dist(p_src)
        q_dist = self.dist(q_src)
        divergence = self.jensen_shannon_divergence(p_dist, q_dist)
        return math.sqrt(divergence)

    def perplexity(self, p_src: str, q_src: str):
        return 2 ** self.cross_entropy(p_src, q_src)

    def cross_entropy_lang(self, lang_dist: dict[str, float], src: str, eps=1e-12):
        q = self.dist(src)
        return sum(q.get(k, eps) * math.log2(1 / max(lang_dist.get(k, eps), eps))
                   for k in self.merge(lang_dist, q))

    def kl_div_lang(self, lang_dist: dict[str, float], src: str, eps=1e-12):
        q = self.dist(src)
        return sum(q.get(k, eps) * math.log2(max(q.get(k, eps), eps) /
                                             max(lang_dist.get(k, eps), eps))
                   for k in self.merge(lang_dist, q))

    def perplexity_lang(self, lang_dist: dict[str, float], src: str, eps=1e-12):
        return 2 ** self.cross_entropy_lang(lang_dist, src, eps)

    def jensen_shannon_distance_lang(self, lang_dist: dict[str, float], src: str, eps=1e-12) -> float:
        q_dist = self.dist(src)
        divergence = self.jensen_shannon_divergence(lang_dist, q_dist, eps=eps)
        return math.sqrt(divergence)

    def conditional_entropy_lang(self, bi_dist, left_dist, src: str, eps=1e-12):
        toks = self.tokens(src)
        if len(toks) < 2:
            return 0.0

        bq = Counter(zip(toks, toks[1:]))
        total = sum(bq.values())
        bq = {k: v / total for k, v in bq.items()}

        pairs = set(bq)
        for a, inner in bi_dist.items():
            for b in inner:
                pairs.add((a, b))

        h = 0.0
        for a, b in pairs:
            q = bq.get((a, b), eps)
            p_cond = bi_dist.get(a, {}).get(b, eps) / left_dist.get(a, eps)
            h += q * math.log2(1 / max(p_cond, eps))
        return h
