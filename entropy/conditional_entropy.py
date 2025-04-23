from common import conditional_entropy


def compute(orig: str, decomp: str) -> float:
    return conditional_entropy(orig, decomp)
