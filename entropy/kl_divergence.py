from .common import kl_div


def compute(orig: str, decomp: str) -> float:
    return kl_div(orig, decomp)
