from entropy.cross_entropy import cross_entropy
from entropy.kl_divergence import kl_div


def entropy(src: str):
    return cross_entropy(src, src)


def nid(p_src: str, q_src: str):
    h_p, h_q = entropy(p_src), entropy(q_src)
    return (kl_div(p_src, q_src) + kl_div(q_src, p_src)) / (h_p + h_q)
