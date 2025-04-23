from common import cross_entropy


def compute(orig: str, decomp: str) -> float:
    return cross_entropy(orig, decomp)
