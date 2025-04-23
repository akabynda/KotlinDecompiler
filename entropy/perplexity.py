from common import perplexity


def compute(orig: str, decomp: str) -> float:
    return perplexity(orig, decomp)
