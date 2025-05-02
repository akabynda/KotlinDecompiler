from entropy.cross_entropy import cross_entropy


def perplexity(p_src: str, q_src: str):
    return 2 ** cross_entropy(p_src, q_src)
